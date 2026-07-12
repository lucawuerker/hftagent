"""Validate + compile LLM-generated executor code (mirrors factor codegen).

Same philosophy as ``agents/factor_research/codegen.py`` + ``factors/inmem.py``:

1. **Sanitise** — reject red-flag tokens (``os``, ``subprocess``, ``open(`` …).
   Not a sandbox; a guardrail against the model going off-script.
2. **Parse + shape** — exactly one ``BaseExecutor`` subclass decorated with
   ``@register_executor``, declaring a matching ``executor_id``, a valid
   ``regime``, a numeric ``params`` dict, and at least one of
   ``step`` / ``target_weights``.
3. **Imports** — allow-list only (numpy / pandas / scipy / sklearn /
   statsmodels + ``quant_fund_agent.execution.base``).  Like factor codegen,
   the allow-list is the LLM arm's *grammar-extension* surface.
4. **In-memory compile** — exec in an isolated namespace, restore the registry
   (transient candidates never shadow persisted executors), optional smoke run
   on a synthetic signal/panel.
"""

from __future__ import annotations

import ast
import logging
import re

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import (
    EXECUTOR_REGISTRY,
    BaseExecutor,
    VALID_REGIMES,
    run_executor,
    validate_weights,
)

log = logging.getLogger("execution.codegen")

_FORBIDDEN_TOKENS = (
    "subprocess", "os.system", "os.popen", "__import__", "eval(", "exec(",
    "open(", "socket", "requests", "urllib", "shutil", "pickle", "pathlib",
    "sys.modules",
)

_ALLOWED_IMPORT_PREFIXES = (
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "sklearn",
    "quant_fund_agent.execution.base",
    "__future__",
)

_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ExecutorValidationError(Exception):
    """Raised when generated executor code fails validation."""


def is_valid_executor_id(executor_id: str) -> bool:
    return bool(_SAFE_ID.match(executor_id))


def _check_forbidden_tokens(code: str) -> None:
    for tok in _FORBIDDEN_TOKENS:
        if tok in code:
            raise ExecutorValidationError(f"forbidden token in executor code: {tok!r}")


def _check_imports(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(_ALLOWED_IMPORT_PREFIXES):
                    raise ExecutorValidationError(
                        f"import of '{alias.name}' is not allowed in executor code")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module.startswith(_ALLOWED_IMPORT_PREFIXES):
                raise ExecutorValidationError(
                    f"import from '{module}' is not allowed in executor code")


def _class_attr(class_def: ast.ClassDef, attr: str) -> ast.AST | None:
    for node in class_def.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == attr:
                    return node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == attr:
                return node.value
    return None


def _literal(node: ast.AST | None):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _find_executor_class(tree: ast.Module, executor_id: str) -> str:
    """Locate the single registered BaseExecutor subclass; return its name."""
    matches: list[ast.ClassDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {getattr(b, "id", getattr(b, "attr", "")) for b in node.bases}
        if "BaseExecutor" in bases:
            matches.append(node)
    if len(matches) != 1:
        raise ExecutorValidationError(
            f"expected exactly one BaseExecutor subclass, found {len(matches)}")
    cls = matches[0]

    decorated = any(
        (isinstance(d, ast.Name) and d.id == "register_executor")
        or (isinstance(d, ast.Attribute) and d.attr == "register_executor")
        for d in cls.decorator_list
    )
    if not decorated:
        raise ExecutorValidationError(
            f"class {cls.name} must be decorated with @register_executor")

    declared = _literal(_class_attr(cls, "executor_id"))
    if declared != executor_id:
        raise ExecutorValidationError(
            f"executor_id mismatch: declared {declared!r}, expected {executor_id!r}")

    regime = _literal(_class_attr(cls, "regime"))
    if regime not in VALID_REGIMES:
        raise ExecutorValidationError(
            f"regime must be one of {VALID_REGIMES}, got {regime!r}")

    params = _literal(_class_attr(cls, "params"))
    if params is not None:
        if not isinstance(params, dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and
            not isinstance(v, bool)
            for k, v in params.items()
        ):
            raise ExecutorValidationError(
                "params must be a dict of numeric constants (the jitter surface)")

    method_names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    if not ({"step", "target_weights"} & method_names):
        raise ExecutorValidationError(
            f"class {cls.name} must implement step() and/or target_weights()")
    return cls.name


def validate_executor_code(code: str, executor_id: str) -> str:
    """Run every static check; return the class name.  Raises on any failure."""
    if not is_valid_executor_id(executor_id):
        raise ExecutorValidationError(f"invalid executor_id: {executor_id!r}")
    _check_forbidden_tokens(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ExecutorValidationError(f"code does not parse: {e}") from e
    _check_imports(tree)
    return _find_executor_class(tree, executor_id)


# ── synthetic smoke fixtures ───────────────────────────────────────────────────

def _make_synthetic_inputs(n_bars: int = 60, n_names: int = 4, seed: int = 7):
    """Tiny deterministic (signal, state, close) trio for the smoke run."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="D")
    cols = [f"T{i}" for i in range(n_names)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, (n_bars, n_names)), axis=0),
        index=idx, columns=cols)
    volume = pd.DataFrame(rng.integers(1_000, 5_000, (n_bars, n_names)).astype(float),
                          index=idx, columns=cols)
    signal = pd.DataFrame(rng.normal(0, 1, (n_bars, n_names)), index=idx, columns=cols)
    panel = {"close": close, "volume": volume}
    from quant_fund_agent.execution.state import build_state_frames

    state = build_state_frames(panel, signal)
    return signal, state, close


def compile_executor_inmem(
    code: str,
    executor_id: str,
    *,
    smoke: bool = True,
) -> type[BaseExecutor]:
    """Validate + exec executor source in-memory; leave the registry untouched.

    Mirrors :func:`quant_fund_agent.factors.inmem.compile_factor`: the
    ``@register_executor`` decorator fires on the byte-identical code that
    would be persisted, then the registry is restored, so transient evolution
    candidates never leak.  ``smoke=True`` additionally runs the program on a
    synthetic signal/panel and validates the output contract — catching runtime
    slips at codegen time, where a retry-with-feedback prompt can fix them.
    """
    class_name = validate_executor_code(code, executor_id)

    prior = EXECUTOR_REGISTRY.pop(executor_id, None)
    namespace: dict = {"__name__": f"<inmem-executor:{executor_id}>"}
    try:
        exec(compile(code, f"<executor:{executor_id}>", "exec"), namespace)  # noqa: S102
        cls = namespace[class_name]
    finally:
        EXECUTOR_REGISTRY.pop(executor_id, None)
        if prior is not None:
            EXECUTOR_REGISTRY[executor_id] = prior

    if not issubclass(cls, BaseExecutor):
        raise ExecutorValidationError(f"{class_name} is not a BaseExecutor subclass")

    if smoke:
        try:
            signal, state, close = _make_synthetic_inputs()
            weights = run_executor(cls(), signal, state, close)
            problems = validate_weights(weights, cls.regime)
        except ExecutorValidationError:
            raise
        except Exception as e:  # noqa: BLE001 — feed straight into the retry prompt
            raise ExecutorValidationError(
                f"executor raised on synthetic data: {e}") from e
        if problems:
            raise ExecutorValidationError(
                f"output contract violated on synthetic data: {problems}")
    return cls
