"""Grammar / render / operator tests for the non-LLM GP factor benchmark.

Covers the three invariants that make the GP a valid, confined benchmark:
* every random/mutated tree renders to code that passes ``validate_code`` and
  computes a non-degenerate signal;
* the GP never reaches an operator outside the base grammar
  (``used_ops(tree) ⊆ ops.BASE_OPS``);
* generation is deterministic under a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund_agent.agents.factor_research.codegen import (
    _make_synthetic_panel,
    validate_code,
)
from quant_fund_agent.agents.factor_research.gp import operators
from quant_fund_agent.agents.factor_research.gp.grammar import (
    OP_BY_NAME,
    build_grammar,
    depth,
    random_tree,
)
from quant_fund_agent.agents.factor_research.gp.render import (
    tree_to_code,
    tree_to_program,
    used_fields,
    used_ops,
)
from quant_fund_agent.factors import inmem
from quant_fund_agent.factors.ops import BASE_OPS

ALLOWED = ["open", "high", "low", "close", "volume", "vwap", "returns"]


@pytest.fixture(scope="module")
def grammar():
    return build_grammar(ALLOWED)


@pytest.fixture(scope="module")
def panel():
    return _make_synthetic_panel()


# ── the grammar is confined to the base grammar ─────────────────────────────────

def test_named_grammar_ops_subset_of_base_ops(grammar):
    named = {op.ops_func for op in grammar.ops if op.ops_func}
    assert named <= BASE_OPS, sorted(named - BASE_OPS)


def test_random_trees_never_reach_ops_outside_base_grammar(grammar):
    rng = np.random.default_rng(0)
    for i in range(300):
        tree = random_tree(grammar, rng, int(rng.integers(2, 7)),
                           method="grow" if i % 2 else "full")
        assert used_ops(tree) <= BASE_OPS


# ── every emitted tree renders + validates + computes ───────────────────────────

def test_random_trees_render_validate_and_compute(grammar, panel):
    rng = np.random.default_rng(1)
    ok = 0
    for i in range(250):
        tree = random_tree(grammar, rng, int(rng.integers(2, 6)),
                           method="grow" if i % 2 else "full")
        fid = f"gp_g_{i:03d}"
        code = tree_to_code(tree, fid, grammar, horizon=6)
        validate_code(code, fid)                       # must not raise
        sig = inmem.signal_from_code(code, fid, panel)  # must not raise
        assert sig.shape == panel["close"].shape
        # inputs are all in scope
        assert used_fields(tree, grammar) <= (set(ALLOWED) | {"close"})
        arr = np.asarray(sig.to_numpy(), dtype="float64")
        finite = np.isfinite(arr)
        if finite.sum() >= 5 and float(np.std(arr[finite])) > 1e-12:
            ok += 1
    # the vast majority of random trees are non-degenerate on real-ish data
    assert ok > 200


def test_depth_bound_respected(grammar):
    rng = np.random.default_rng(2)
    for _ in range(200):
        d = int(rng.integers(1, 6))
        tree = random_tree(grammar, rng, d, method="full")
        assert depth(tree) <= d


# ── operators preserve validity / typing / depth cap ────────────────────────────

def test_operators_preserve_validity_and_depth(grammar, panel):
    rng = np.random.default_rng(3)
    for _ in range(80):
        a = random_tree(grammar, rng, 4, method="grow")
        b = random_tree(grammar, rng, 4, method="grow")
        children = {
            "crossover": operators.subtree_crossover(a, b, grammar, rng, 6),
            "subtree": operators.subtree_mutation(a, grammar, rng, 6),
            "point": operators.point_mutation(a, grammar, rng),
            "hoist": operators.hoist_mutation(a, rng),
        }
        for name, child in children.items():
            assert depth(child) <= 6, name
            assert used_ops(child) <= BASE_OPS, name
            fid = f"gp_op_{name}"
            code = tree_to_code(child, fid, grammar, horizon=6)
            validate_code(code, fid)
            inmem.signal_from_code(code, fid, panel)


def test_point_mutation_keeps_operator_signature(grammar):
    """An op is only ever swapped for one with an identical arg signature."""
    rng = np.random.default_rng(7)
    # a correlation(series, series, window) node
    corr = OP_BY_NAME["correlation"]
    from quant_fund_agent.agents.factor_research.gp.grammar import Node

    tree = Node("op", "correlation", (
        Node("field", "close"), Node("field", "volume"), Node("window", 10)))
    for _ in range(50):
        mutated = operators.point_mutation(tree, grammar, rng)
        # root op (if still an op) must keep the 3-arg (S,S,W) signature
        if mutated.kind == "op":
            assert OP_BY_NAME[mutated.value].arg_types == corr.arg_types


# ── determinism ─────────────────────────────────────────────────────────────────

def test_generation_is_deterministic(grammar):
    t1 = random_tree(grammar, np.random.default_rng(42), 5, method="grow")
    t2 = random_tree(grammar, np.random.default_rng(42), 5, method="grow")
    assert t1.to_dict() == t2.to_dict()


# ── data-scope gating: grammar only draws in-scope fields ───────────────────────

def test_grammar_confined_to_allowed_fields():
    grammar = build_grammar(["close", "volume"])  # a restricted scope
    rng = np.random.default_rng(5)
    for _ in range(100):
        tree = random_tree(grammar, rng, 4, method="grow")
        prog = tree_to_program(tree, "gp_scope", grammar, horizon=6)
        cls = inmem.compile_factor(prog.code, "gp_scope")
        assert set(cls.inputs) <= {"close", "volume"}


def test_build_grammar_rejects_empty_scope():
    with pytest.raises(ValueError):
        build_grammar([])
