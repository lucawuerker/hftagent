"""Tests for the Pareto objective vector, gates and dominance primitives (P0)."""

from __future__ import annotations

import numpy as np

from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
    complexity,
    dominates,
    non_dominated_front,
    participation_ratio,
)


def _result(cid, mv, indep, rob, par, *, passed=True):
    obj = ObjectiveVector(marginal_value=mv, independence=indep, robustness=rob, parsimony=par)
    gates = GateResults(coverage_ok=passed, degradation_ok=passed, deflation_ok=passed)
    return FitnessResult(candidate_id=cid, objective=obj, gates=gates)


# ── objective vector ──────────────────────────────────────────────────────────

def test_objective_none_is_worst():
    v = ObjectiveVector(marginal_value=None, independence=1.0, robustness=2.0, parsimony=-3.0)
    t = v.as_tuple()
    assert t[0] == float("-inf")
    assert t[1:] == (1.0, 2.0, -3.0)


# ── dominance ─────────────────────────────────────────────────────────────────

def test_dominance_basic():
    a = ObjectiveVector(1.0, 1.0, 1.0, 1.0)
    b = ObjectiveVector(0.5, 1.0, 1.0, 1.0)     # strictly worse on axis 1, tied else
    c = ObjectiveVector(2.0, 0.0, 1.0, 1.0)     # better on 1, worse on 2 → non-comparable
    assert dominates(a, b)
    assert not dominates(b, a)
    assert not dominates(a, c) and not dominates(c, a)


def test_gates_passed_semantics():
    assert GateResults(True, True, True).passed
    assert GateResults(True, None, True).passed          # None = not evaluated, ok
    assert not GateResults(True, False, True).passed


# ── non-dominated front honours the gates ─────────────────────────────────────

def test_front_prefers_gate_passers():
    good = _result("g", 1.0, 1.0, 1.0, -1.0, passed=True)
    dominated = _result("d", 0.1, 0.1, 0.1, -5.0, passed=True)
    # a gate-failing candidate with a great objective must NOT make the front while
    # a gate-passing candidate exists
    cheater = _result("c", 5.0, 5.0, 5.0, -0.0, passed=False)
    front = non_dominated_front([good, dominated, cheater])
    ids = {r.candidate_id for r in front}
    assert "g" in ids and "c" not in ids and "d" not in ids


def test_front_falls_back_when_none_pass():
    a = _result("a", 1.0, 0.0, 0.0, 0.0, passed=False)
    b = _result("b", 0.0, 1.0, 0.0, 0.0, passed=False)
    c = _result("c", 0.0, 0.0, 0.0, 0.0, passed=False)  # dominated by both a and b
    front = non_dominated_front([a, b, c])
    ids = {r.candidate_id for r in front}
    assert ids == {"a", "b"}


# ── participation ratio ───────────────────────────────────────────────────────

def test_participation_ratio_orthogonal_vs_identical():
    ortho = np.eye(4)
    assert participation_ratio(ortho) == 4.0
    identical = np.ones((4, 4))
    assert participation_ratio(identical) < 1.5     # one effective direction


# ── AST complexity ────────────────────────────────────────────────────────────

def test_complexity_counts_ops_and_constants():
    simple = "def calc(self):\n    return self.close"
    complex_ = "def calc(self):\n    return (self.close - self.open) / (self.high + 2) * 3"
    assert complexity(simple) < complexity(complex_)
    assert complexity("") == 0
    assert complexity("def (bad syntax") == 10_000
