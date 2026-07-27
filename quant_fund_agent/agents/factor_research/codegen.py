"""Validate, write, and load LLM-generated factor code.

The Factor Researcher Agent asks an LLM for full Python source for a
``BaseFactor`` subclass.  Before we let that code into the runtime
registry we have to:

1.  **Sanitise**: reject obvious red flags (``os``, ``subprocess``,
    ``open(``, ``eval(``, ``exec(``, ``__import__``).  This is *not* a
    real sandbox — the LLM is on our side — but it catches the cases
    where the model goes off-script.
2.  **Parse**: ``ast.parse`` the code.  Reject if it doesn't parse.
3.  **Verify shape**: it must define exactly one class subclassing
    ``BaseFactor`` and declare a non-empty ``factor_id`` matching the
    spec.
4.  **Import**: write the file inside the ``factors/researcher/``
    package and import it via ``importlib``.  This triggers the
    ``@register_factor`` decorator and inserts the class into the
    global factor registry.
5.  **Smoke test**: instantiate the class and call ``calc`` on a small
    synthetic OHLCV panel to make sure it doesn't blow up.

Failing candidates are dropped — the rest of the agent's pipeline
continues without them.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from quant_fund_agent.agents.factor_research.prompts import MAX_PREDICTION_HORIZON
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import get_factor_class

log = logging.getLogger("factor_research.codegen")

# Module path where generated files live so the existing
# ``discover_factors`` mechanism picks them up.
RESEARCHER_PKG = "quant_fund_agent.factors.researcher"
RESEARCHER_DIR = Path(__file__).resolve().parents[2] / "factors" / "researcher"

# Tokens we never want to see in LLM-generated code.  This is not a
# real sandbox — it's a guardrail against the model going off-script.
_FORBIDDEN_TOKENS = (
    "subprocess",
    "os.system",
    "os.popen",
    "__import__",
    "eval(",
    "exec(",
    "open(",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pickle",
    "pathlib",
    "sys.modules",
)

# Imports that *are* allowed in researcher code.  Anything else and we
# bail out.  The scientific-computing stack (numpy/scipy/statsmodels) is the
# deliberate surface the LLM researcher implements *its own* paper maths on
# (signature transforms, Hawkes intensities, spectral / entropy features,
# rolling regressions, …) — it is intentionally broader than the fixed
# ``BASE_OPS`` grammar the non-LLM GP benchmark is confined to.  The stdlib
# compute modules below (``math``, ``typing``, …) are pure and side-effect
# free; they are allowed because the model reaches for them by reflex and
# rejecting them only burns a retry.  I/O / system modules stay out (and are
# also caught by ``_FORBIDDEN_TOKENS``).
_ALLOWED_IMPORT_PREFIXES = (
    # scientific-computing stack — implement richer paper maths here
    "pandas",
    "numpy",
    "scipy",
    "statsmodels",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.preprocessing",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.naive_bayes",
    # pure-stdlib compute helpers (no I/O, no system access)
    "math",
    "cmath",
    "statistics",
    "functools",
    "itertools",
    "collections",
    "dataclasses",
    "typing",
    "numbers",
    "fractions",
    "warnings",
    # the factor framework
    "quant_fund_agent.factors.base",
    "quant_fund_agent.factors.registry",
    "quant_fund_agent.factors.ops",
    "__future__",
)


# Legal names you can import from ``quant_fund_agent.factors.ops``.
# Derived from the module itself so this list never goes stale.  Used
# to bounce the very common LLM mistake of trying to import a pandas
# DataFrame method (``fillna``, ``where``, ``replace``) from ``ops``,
# which would later fail with an unhelpful ``ImportError``.
def _legal_ops_names() -> frozenset[str]:
    from quant_fund_agent.factors import ops as _ops_mod
    return frozenset(
        name for name in dir(_ops_mod)
        if not name.startswith("_") and callable(getattr(_ops_mod, name))
    )


_LEGAL_OPS = _legal_ops_names()
_OPS_MODULE = "quant_fund_agent.factors.ops"


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def is_valid_factor_id(factor_id: str) -> bool:
    """Reject anything that isn't a sensible Python identifier-ish slug."""
    return bool(_SAFE_ID.match(factor_id))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class CodeValidationError(Exception):
    """Raised when generated factor code fails validation."""


def _check_forbidden_tokens(code: str) -> None:
    for tok in _FORBIDDEN_TOKENS:
        if tok in code:
            raise CodeValidationError(f"forbidden token in generated code: {tok!r}")


def _check_imports(tree: ast.Module) -> None:
    """Reject (a) any import outside the allow-list, and (b) any
    ``from quant_fund_agent.factors.ops import X`` where X is not an
    actual op.

    (b) is the rule that catches the LLM's most common slip: trying to
    import pandas DataFrame methods (``fillna``, ``where``, ``replace``,
    …) from ``ops`` as if they were free functions.  Catching this
    here — with a precise error message naming the bad symbol — gives
    the retry prompt something it can act on, instead of letting it
    blow up later as a generic ``ImportError``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith(_ALLOWED_IMPORT_PREFIXES):
                    raise CodeValidationError(
                        f"disallowed import: {alias.name!r}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if not module.startswith(_ALLOWED_IMPORT_PREFIXES):
                raise CodeValidationError(
                    f"disallowed `from {module} import …`"
                )
            if module == _OPS_MODULE:
                for alias in node.names:
                    if alias.name == "*":
                        raise CodeValidationError(
                            f"`from {_OPS_MODULE} import *` is not allowed; "
                            "import the operators you need by name"
                        )
                    if alias.name not in _LEGAL_OPS:
                        raise CodeValidationError(
                            f"{alias.name!r} is not defined in "
                            f"`{_OPS_MODULE}`.  If you need it, it is "
                            f"almost certainly a pandas DataFrame method "
                            f"(e.g. ``df.{alias.name}(...)``) or a numpy "
                            f"function — call it directly, do not import "
                            f"it from the ops module."
                        )


def _is_negative_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and node.operand.value > 0
    )


def _is_true_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _check_temporal_leakage(tree: ast.Module) -> None:
    """Reject common future-looking constructs in generated factor code.

    This is deliberately a small static net, not a proof of causality.  It catches
    the patterns LLMs most often reach for when they accidentally turn the target
    into a feature: negative shifts/diffs/pct-changes, centered rolling windows,
    and fitting a learner inside ``calc`` on the full panel.
    """
    future_attrs = {"shift", "diff", "pct_change"}
    fit_attrs = {"fit", "fit_transform", "partial_fit"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        attr = func.attr

        if attr in future_attrs:
            if node.args and _is_negative_literal(node.args[0]):
                raise CodeValidationError(
                    f"look-ahead risk: `{attr}` called with a negative period"
                )
            for kw in node.keywords:
                if kw.arg in {"periods", "lag"} and _is_negative_literal(kw.value):
                    raise CodeValidationError(
                        f"look-ahead risk: `{attr}` called with negative {kw.arg}"
                    )

        if attr == "rolling":
            for kw in node.keywords:
                if kw.arg == "center" and _is_true_literal(kw.value):
                    raise CodeValidationError(
                        "look-ahead risk: centered rolling windows are not allowed"
                    )

        if attr in fit_attrs:
            raise CodeValidationError(
                f"look-ahead risk: `{attr}()` inside a factor can fit on the full "
                "panel.  Use deterministic trailing transforms in calc(); learned "
                "models need an explicit fit/predict interface."
            )


def _extract_class_string_list(class_def: ast.ClassDef, attr: str) -> list[str] | None:
    """Read ``ClassName.attr = [\"x\", \"y\"]`` from a ClassDef body.

    Returns ``None`` if the attribute is not declared (or not a list of
    string literals).  Used to read ``inputs = [...]`` from generated
    factor classes.
    """
    for stmt in class_def.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == attr
            and isinstance(stmt.value, ast.List)
        ):
            try:
                return [el.value for el in stmt.value.elts if isinstance(el, ast.Constant)]
            except Exception:
                return None
    return None


def _extract_class_int(class_def: ast.ClassDef, attr: str) -> int | None:
    """Read ``ClassName.attr = <int>`` from a ClassDef body.

    Handles a plain int literal and a negated literal (``-5`` →
    ``UnaryOp(USub, …)``) so the caller can report "must be positive" rather
    than "missing".  Returns ``None`` if the attribute is absent or not an int
    literal.  Used to read ``prediction_horizon`` from generated factor classes.
    """
    for stmt in class_def.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == attr
        ):
            val = stmt.value
            if isinstance(val, ast.Constant) and isinstance(val.value, int) \
                    and not isinstance(val.value, bool):
                return val.value
            if (
                isinstance(val, ast.UnaryOp)
                and isinstance(val.op, ast.USub)
                and isinstance(val.operand, ast.Constant)
                and isinstance(val.operand.value, int)
            ):
                return -val.operand.value
            return None
    return None


def _extract_class_int_list(class_def: ast.ClassDef, attr: str) -> list[int] | None:
    """Read ``ClassName.attr = [1, 6, 60]`` from a ClassDef body.

    Returns the list of int-literal element values (``None`` if the attribute is
    absent or not a list).  Non-int elements are kept as ``None`` so the caller
    can flag them.  Used to read ``suggested_horizons``.
    """
    for stmt in class_def.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == attr
            and isinstance(stmt.value, ast.List)
        ):
            out: list[int] = []
            for el in stmt.value.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, int) \
                        and not isinstance(el.value, bool):
                    out.append(el.value)
                else:
                    out.append(None)  # type: ignore[arg-type]
            return out
    return None


def _data_field_references(class_def: ast.ClassDef) -> set[str]:
    """Collect every constant string ``X`` used as ``data["X"]`` inside
    the class body.

    This is the "what does this factor actually touch" set, derived
    from the source.  We use it to enforce that the declared
    ``inputs = [...]`` covers every field the code reads from ``data``,
    catching the very common LLM mistake of writing ``data["lobImb"]``
    but forgetting to put ``"lobImb"`` in ``inputs``.
    """
    fields: set[str] = set()
    for node in ast.walk(class_def):
        if not isinstance(node, ast.Subscript):
            continue
        # Match exactly: data["X"]
        value = node.value
        if not (isinstance(value, ast.Name) and value.id == "data"):
            continue
        # Python 3.9+: slice is the expression directly
        slice_expr = node.slice
        if isinstance(slice_expr, ast.Constant) and isinstance(slice_expr.value, str):
            fields.add(slice_expr.value)
    return fields


def _find_factor_class(
    tree: ast.Module,
    expected_factor_id: str,
    expected_prediction_horizon: int | None = None,
) -> str:
    """Return the (single) class name that subclasses BaseFactor.

    In addition to checking the class shape and ``factor_id`` match,
    this also enforces:
      - ``inputs = [...]`` is declared and non-empty,
      - every field referenced as ``data[\"X\"]`` inside the class is
        listed in ``inputs``.

    These two rules matter because the agent uses each candidate's
    ``inputs`` attribute to decide which fields to materialise on the
    real panel.  An undeclared field means the panel will be missing
    that column and the backtest will crash with ``KeyError`` — a
    failure mode that's much more expensive to discover at backtest
    time than at validation time.
    """
    candidates: list[ast.ClassDef] = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            (b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", ""))
            for b in node.bases
        }
        if "BaseFactor" not in base_names:
            continue
        # Check that factor_id is assigned correctly.
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "factor_id"
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value == expected_factor_id
            ):
                candidates.append(node)
                break
    if len(candidates) != 1:
        raise CodeValidationError(
            f"expected exactly one BaseFactor subclass with factor_id="
            f"{expected_factor_id!r}, found {len(candidates)}"
        )

    cls = candidates[0]

    declared_inputs = _extract_class_string_list(cls, "inputs")
    if declared_inputs is None:
        raise CodeValidationError(
            f"class {cls.name!r} is missing a class attribute "
            f"`inputs = [\"field1\", \"field2\", ...]`.  This list "
            f"tells the agent which data fields to load for the "
            f"backtest, so it MUST enumerate every field accessed via "
            f"``data[\"...\"]`` inside calc()."
        )
    if not declared_inputs:
        raise CodeValidationError(
            f"class {cls.name!r} declares `inputs = []`.  At least one "
            f"data field is required."
        )

    referenced = _data_field_references(cls)
    missing = referenced - set(declared_inputs)
    if missing:
        raise CodeValidationError(
            f"class {cls.name!r} reads {sorted(missing)} from ``data`` "
            f"but does not list them in `inputs = {declared_inputs}`.  "
            f"Add the missing field(s) to the `inputs` class attribute "
            f"so the agent loads them onto the panel."
        )

    horizon = _extract_class_int(cls, "prediction_horizon")
    if horizon is None:
        raise CodeValidationError(
            f"class {cls.name!r} is missing a positive-int class attribute "
            f"`prediction_horizon = <bars>` — the forward offset (in bars) at "
            f"which the factor's edge is expected to peak.  Declare it like the "
            f"seed factors do."
        )
    if horizon <= 0 or horizon > MAX_PREDICTION_HORIZON:
        raise CodeValidationError(
            f"class {cls.name!r} has prediction_horizon={horizon}; it must be a "
            f"positive integer ≤ {MAX_PREDICTION_HORIZON} bars."
        )
    if (
        expected_prediction_horizon is not None
        and horizon != int(expected_prediction_horizon)
    ):
        raise CodeValidationError(
            f"class {cls.name!r} has prediction_horizon={horizon}; this run fixes "
            f"the researcher horizon at {int(expected_prediction_horizon)} bars, "
            f"so declare `prediction_horizon = {int(expected_prediction_horizon)}`."
        )

    suggested = _extract_class_int_list(cls, "suggested_horizons")
    if suggested is not None:
        bad = [h for h in suggested
               if h is None or h <= 0 or h > MAX_PREDICTION_HORIZON]
        if bad:
            raise CodeValidationError(
                f"class {cls.name!r} has an invalid `suggested_horizons` entry "
                f"{bad}; every element must be a positive integer ≤ "
                f"{MAX_PREDICTION_HORIZON} bars."
            )

    return cls.name


def validate_code(
    code: str,
    expected_factor_id: str,
    expected_prediction_horizon: int | None = None,
) -> str:
    """Run all static checks, return the discovered class name."""
    if not code or not code.strip():
        raise CodeValidationError("empty code")
    if not is_valid_factor_id(expected_factor_id):
        raise CodeValidationError(f"invalid factor_id: {expected_factor_id!r}")
    _check_forbidden_tokens(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"syntax error: {e}") from e
    _check_imports(tree)
    _check_temporal_leakage(tree)
    return _find_factor_class(tree, expected_factor_id, expected_prediction_horizon)


# ---------------------------------------------------------------------------
# File emission + dynamic import
# ---------------------------------------------------------------------------

def _module_name_for(factor_id: str) -> str:
    return f"{RESEARCHER_PKG}.{factor_id}"


def write_factor_file(factor_id: str, code: str) -> Path:
    """Write the LLM-generated code to ``factors/researcher/<id>.py``."""
    RESEARCHER_DIR.mkdir(parents=True, exist_ok=True)
    path = RESEARCHER_DIR / f"{factor_id}.py"
    path.write_text(code, encoding="utf-8")
    return path


def import_factor_module(factor_id: str) -> None:
    """Import (or reload) the researcher module so ``@register_factor`` fires."""
    module_name = _module_name_for(factor_id)
    if module_name in importlib.sys.modules:
        importlib.reload(importlib.sys.modules[module_name])
    else:
        importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_synthetic_panel(
    n_bars: int = 200,
    n_tickers: int = 6,
) -> dict[str, pd.DataFrame]:
    """Tiny panel matching the live data loader's field set.

    Includes every field ``load_panel`` produces (OHLCV view + every
    raw LOBSTER passthrough), so factors that reference signed trades,
    order flow, effective spread, event counts, etc. can run during
    the smoke test without KeyError-ing.
    """
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="10s")
    cols = [f"T{i}" for i in range(n_tickers)]

    def _df(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=idx, columns=cols)

    close = _df(100 + np.cumsum(rng.normal(0, 0.01, (n_bars, n_tickers)), axis=0))
    open_ = close.shift(1).fillna(close)
    high = pd.concat([open_, close], axis=0).groupby(level=0).max().reindex(idx)
    low = pd.concat([open_, close], axis=0).groupby(level=0).min().reindex(idx)

    trade = _df(rng.integers(-1000, 1000, (n_bars, n_tickers)).astype(float))
    volume = trade.abs()

    return {
        # OHLCV view
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        # raw LOBSTER passthroughs
        "trade": trade,
        "orderFlow": _df(rng.integers(-2000, 2000, (n_bars, n_tickers)).astype(float)),
        "hidden": _df(rng.integers(0, 200, (n_bars, n_tickers)).astype(float)),
        "auction": _df(np.zeros((n_bars, n_tickers))),
        "spread": _df(rng.uniform(0.01, 0.05, (n_bars, n_tickers))),
        "effSpread": _df(rng.uniform(0.005, 0.05, (n_bars, n_tickers))),
        "lobImb": _df(rng.uniform(-1, 1, (n_bars, n_tickers))),
        "effLobImb": _df(rng.uniform(-1, 1, (n_bars, n_tickers))),
        "trdLiq": _df(rng.uniform(0, 1, (n_bars, n_tickers))),
        "ofLiq": _df(rng.uniform(0, 1, (n_bars, n_tickers))),
        "depth": _df(rng.integers(100, 10_000, (n_bars, n_tickers)).astype(float)),
        "nbEvents": _df(rng.integers(0, 50, (n_bars, n_tickers)).astype(float)),
        "nbHidden": _df(rng.integers(0, 5, (n_bars, n_tickers)).astype(float)),
        "nbTrades": _df(rng.integers(0, 20, (n_bars, n_tickers)).astype(float)),
    }


def smoke_test(factor_id: str) -> None:
    """Instantiate the registered factor and run ``calc`` on toy data."""
    cls = get_factor_class(factor_id)
    if cls is None:
        raise CodeValidationError(
            f"factor {factor_id!r} did not register itself "
            "(missing @register_factor or wrong factor_id?)"
        )
    if not issubclass(cls, BaseFactor):
        raise CodeValidationError(f"{cls.__name__} is not a BaseFactor subclass")
    instance = cls()
    data = _make_synthetic_panel()
    try:
        out = instance.calc(data)
    except Exception as e:
        raise CodeValidationError(f"calc() raised on synthetic data: {e}") from e
    if not isinstance(out, pd.DataFrame):
        raise CodeValidationError(
            f"calc() must return a DataFrame, got {type(out).__name__}"
        )
    if out.shape != data["close"].shape:
        raise CodeValidationError(
            f"calc() output shape {out.shape} != close shape {data['close'].shape}"
        )


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def materialise(
    factor_id: str,
    code: str,
    expected_prediction_horizon: int | None = None,
) -> Path:
    """Validate, write, import, smoke-test.  Returns the file path on success.

    On any post-write failure (import error, smoke-test crash) we:
      1. delete the file so ``discover_factors`` won't pick up dead code,
      2. drop any partial entry from the in-memory factor registry so the
         next attempt with the same ``factor_id`` is not blocked by
         ``register_factor`` raising "Duplicate factor_id".  This matters
         for the agent's retry-on-failure path: the LLM's first attempt
         can register the class via ``@register_factor`` before the smoke
         test crashes, leaving a stale registry entry behind.
    """
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY, get_factor_class

    class_name = validate_code(code, factor_id, expected_prediction_horizon)
    log.info("validated %s (class %s)", factor_id, class_name)
    # Never overwrite an already-registered factor's code (e.g. another prerun's
    # under dedup_scope="prerun"): fail cleanly so the clash is dropped instead of
    # clobbering and then unlinking a file we don't own.
    if get_factor_class(factor_id) is not None:
        raise CodeValidationError(
            f"factor_id {factor_id!r} already exists in the registry; skipping "
            f"to avoid overwriting another factor's code"
        )
    path = write_factor_file(factor_id, code)
    try:
        import_factor_module(factor_id)
        smoke_test(factor_id)
    except Exception:
        path.unlink(missing_ok=True)
        _FACTOR_REGISTRY.pop(factor_id, None)
        importlib.sys.modules.pop(_module_name_for(factor_id), None)
        raise
    return path


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def purge_researcher_code() -> int:
    """Delete every generated factor file (keeps __init__.py)."""
    if not RESEARCHER_DIR.exists():
        return 0
    n = 0
    for f in RESEARCHER_DIR.glob("*.py"):
        if f.name == "__init__.py":
            continue
        f.unlink()
        n += 1
    return n
