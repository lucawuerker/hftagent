"""Guard: every arm in ``matrix/final_matrix.yaml`` must produce an argv its
target entrypoint's argparse actually accepts.

The orchestrator builds each run's command line from YAML (defaults + per-arm
flags), and each entrypoint accepts a *different* flag vocabulary — evolution
flags leaking into the GP/oneshot argv are an immediate ``argparse`` crash that
would otherwise only surface mid-matrix, after credits were already spent on
earlier arms.  This test would have caught exactly that: it AST-extracts the
``add_argument`` specs from each ``run_*.py`` (no entrypoint imports — they read
the environment at import time) and walks each arm's generated argv against
them: unknown flags, values handed to ``store_true`` flags, flags missing their
value, and out-of-``choices`` values all fail here instead of on the cluster.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO / "matrix" / "final_matrix.yaml"

run_ablation_matrix = importlib.import_module("run_ablation_matrix")

ENTRY_SCRIPT = {
    "evolution": "run_factor_evolution.py",
    "gp": "run_gp_factor_mining.py",
    "oneshot": "run_factor_research.py",
}


def _parser_spec(script: Path) -> dict[str, dict]:
    """``--flag -> {store_true, choices}`` for every add_argument in *script*."""
    spec: dict[str, dict] = {}
    for node in ast.walk(ast.parse(script.read_text())):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        options = [a.value for a in node.args
                   if isinstance(a, ast.Constant) and isinstance(a.value, str)
                   and a.value.startswith("--")]
        if not options:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        action = kw.get("action")
        store_true = (isinstance(action, ast.Constant)
                      and action.value == "store_true")
        choices = None
        ch = kw.get("choices")
        if isinstance(ch, (ast.List, ast.Tuple)) and \
                all(isinstance(e, ast.Constant) for e in ch.elts):
            choices = [str(e.value) for e in ch.elts]
        for opt in options:
            spec[opt] = {"store_true": store_true, "choices": choices}
    return spec


@pytest.fixture(scope="module")
def plan() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text())


@pytest.fixture(scope="module")
def parser_specs() -> dict[str, dict[str, dict]]:
    return {entry: _parser_spec(REPO / script)
            for entry, script in ENTRY_SCRIPT.items()}


def test_plan_structure(plan):
    for key in ("config_file", "providers", "arms"):
        assert key in plan
    assert (REPO / plan["config_file"]).exists()
    names = [a["name"] for a in plan["arms"]]
    assert len(names) == len(set(names)), "duplicate arm names"
    for arm in plan["arms"]:
        prov = arm.get("provider")
        if prov is not None:
            assert prov in plan["providers"], \
                f"arm {arm['name']} references unknown provider {prov!r}"


def test_fixed_and_reference_books_exist(plan):
    for key in ("fixed-book", "reference-book"):
        path = plan.get("defaults", {}).get(key)
        if path:
            assert (REPO / path).exists(), f"defaults.{key} missing: {path}"


def test_every_arm_argv_parses_against_its_entrypoint(plan, parser_specs):
    for arm in plan["arms"]:
        entry = arm.get("entrypoint", "evolution")
        argv, _env, _name = run_ablation_matrix.arm_command(plan, arm)
        assert Path(argv[1]).name == ENTRY_SCRIPT[entry]
        spec = parser_specs[entry]

        tokens = argv[2:]
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            assert tok.startswith("--"), \
                f"{arm['name']}: stray positional {tok!r} in {tokens}"
            assert tok in spec, \
                f"{arm['name']}: {ENTRY_SCRIPT[entry]} does not accept {tok!r}"
            if spec[tok]["store_true"]:
                i += 1
                continue
            assert i + 1 < len(tokens) and not tokens[i + 1].startswith("--"), \
                f"{arm['name']}: flag {tok} is missing its value"
            value = tokens[i + 1]
            choices = spec[tok]["choices"]
            if choices is not None:
                assert value in choices, \
                    f"{arm['name']}: {tok}={value!r} not in choices {choices}"
            i += 2


def test_evolution_defaults_do_not_leak_into_gp_or_oneshot(plan):
    """The regression this file exists for: L0/L1 arms must never inherit the
    evolution ``defaults`` block."""
    evolution_only = {"--retrieval", "--mechanism-groups", "--progressive-reveal",
                      "--archive-cap", "--creative-frac", "--fixed-book",
                      "--debate", "--max-cost-usd"}
    for arm in plan["arms"]:
        if arm.get("entrypoint", "evolution") == "evolution":
            continue
        argv, _env, _name = run_ablation_matrix.arm_command(plan, arm)
        leaked = evolution_only & set(argv)
        assert not leaked, f"{arm['name']} inherited evolution flags: {leaked}"
