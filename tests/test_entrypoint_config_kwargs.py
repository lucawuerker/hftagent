"""Guard: every ``run_*.py`` entrypoint must construct its config dataclasses with
keyword arguments that still exist.

Entrypoints are argparse-only shims, so nothing imports them and no test used to
execute them — which let them rot silently.  When the Pareto vector lost the CPCV
/ regime / QD machinery, ``run_gp_factor_mining.py`` kept passing twelve removed
kwargs and died with a ``TypeError`` on the first line of real work; the timing
harness kept passing two.  Both bugs are invisible to every other test in the
suite because the loops they drive are perfectly healthy.

This test parses each entrypoint (no execution — importing them is not free and
some read the environment at import time), finds every call to a dataclass it
imports, and asserts the keywords are real fields.  It is deliberately generic:
any future field rename in a config dataclass fails here rather than in a
half-hour run.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = sorted(REPO_ROOT.glob("run_*.py"))


def _imported_names(tree: ast.Module) -> dict[str, str]:
    """Map locally-bound name → the module it was imported from.

    Covers both top-level and function-local ``from x import y`` (the entrypoints
    import lazily inside ``main()`` so the data config loads before the panel).
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bound[alias.asname or alias.name] = node.module
    return bound


def _dataclass_calls(tree: ast.Module, bound: dict[str, str]):
    """Yield ``(lineno, name, dataclass, keywords)`` for calls to imported dataclasses."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        module = bound.get(name)
        if module is None or not module.startswith("quant_fund_agent"):
            continue
        try:
            obj = getattr(importlib.import_module(module), name, None)
        except Exception:  # pragma: no cover - an unimportable module is another test's job
            continue
        if obj is None or not dataclasses.is_dataclass(obj):
            continue
        yield node.lineno, name, obj, [kw.arg for kw in node.keywords if kw.arg]


@pytest.mark.parametrize("path", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_passes_only_real_config_fields(path: Path) -> None:
    tree = ast.parse(path.read_text())
    bound = _imported_names(tree)

    problems: list[str] = []
    for lineno, name, obj, kwargs in _dataclass_calls(tree, bound):
        fields = {f.name for f in dataclasses.fields(obj)}
        dead = [k for k in kwargs if k not in fields]
        if dead:
            problems.append(
                f"{path.name}:{lineno} {name}(...) passes removed field(s): "
                f"{', '.join(sorted(dead))}")

    assert not problems, "stale entrypoint config kwargs:\n  " + "\n  ".join(problems)


def test_the_guard_actually_inspects_something() -> None:
    """A typo in the AST walk would make every case above vacuously pass."""
    gp = REPO_ROOT / "run_gp_factor_mining.py"
    tree = ast.parse(gp.read_text())
    calls = list(_dataclass_calls(tree, _imported_names(tree)))
    assert any(name == "GPRunConfig" and kwargs for _, name, _, kwargs in calls), (
        "expected to find the GPRunConfig(...) construction in "
        "run_gp_factor_mining.py — the walk is not resolving lazy imports")
