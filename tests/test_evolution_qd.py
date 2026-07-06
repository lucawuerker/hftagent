"""Tests for the Quality-Diversity behavior grid (WS2).

The QD archive decouples *diversity of search* (the behavior grid) from *quality*
(the unchanged 5-axis Pareto).  These tests exercise binning, the capped mini-Pareto
per cell, cell/elite sampling, determinism, the P7 reuse bias, save/load (frozen bin
edges), and the controller QD-mode wiring — all without market data.
"""

from __future__ import annotations

import numpy as np

from quant_fund_agent.agents.factor_research.evolution.controller import (
    ControllerConfig,
    EvaluatedGenome,
    EvolutionController,
)
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram, Genome
from quant_fund_agent.agents.factor_research.evolution.qd import (
    QDArchive,
    QDConfig,
    _bucket,
)
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
)


def _eg(gid, behavior, *, mv=0.1, passed=True) -> EvaluatedGenome:
    gates = (GateResults(coverage_ok=True, degradation_ok=True)
             if passed else GateResults(coverage_ok=False))
    fit = FitnessResult(
        candidate_id=gid,
        objective=ObjectiveVector(marginal_value=mv, independence=0.0,
                                  robustness=0.0, parsimony=0.0),
        gates=gates, behavior=behavior)
    genome = Genome(genome_id=gid,
                    programs=[FactorProgram(factor_id=gid, code=f"x = {gid!r}\n")])
    return EvaluatedGenome(genome=genome, fitness=fit)


# ── binning ───────────────────────────────────────────────────────────────────

def test_bucket_fixed_edges_and_nonfinite_fallback():
    edges = (-0.33, 0.33)
    assert _bucket(-0.5, edges) == 0      # fade
    assert _bucket(0.0, edges) == 1       # neutral
    assert _bucket(0.5, edges) == 2       # momentum
    assert _bucket(None, edges) == 0      # missing → fallback bin
    assert _bucket(float("nan"), edges) == 0


def test_cell_key_2d_and_3d():
    a2 = QDArchive(QDConfig(grid_dims=2))
    assert a2.cell_key({"trend_reversal": -0.5, "signal_speed": 0.2}) == (0, 0)
    assert a2.cell_key({"trend_reversal": 0.5, "signal_speed": 1.5}) == (2, 2)
    a3 = QDArchive(QDConfig(grid_dims=3))
    k = a3.cell_key({"trend_reversal": 0.0, "signal_speed": 0.7,
                     "stress_activation": 1.5})
    assert len(k) == 3 and k[2] == 2      # 1.5 ≥ 1.3 → activated


# ── cell competition (capped mini-Pareto) ─────────────────────────────────────

def test_cell_capacity_keeps_the_best_by_pareto():
    arch = QDArchive(QDConfig(cell_capacity=3))
    b = {"trend_reversal": 0.0, "signal_speed": 0.7}   # all land in ONE cell
    for i, mv in enumerate([0.1, 0.5, 0.3, 0.9, 0.2]):
        arch.insert(_eg(f"g{i}", b, mv=mv))
    assert arch.n_cells() == 1
    cell = next(iter(arch.cells.values()))
    assert len(cell) == 3
    # objectives differ only on marginal_value → total dominance order → top-3 kept
    kept = sorted(eg.fitness.objective.marginal_value for eg in cell)
    assert kept == [0.3, 0.5, 0.9]


def test_gate_failer_never_occupies_a_cell():
    arch = QDArchive()
    assert arch.insert(_eg("f", {"trend_reversal": 0.0, "signal_speed": 0.5},
                            passed=False)) is None
    assert arch.n_cells() == 0


def test_nonfinite_behavior_falls_back_to_bin_zero_no_crash():
    arch = QDArchive()
    arch.insert(_eg("x", {"trend_reversal": None, "signal_speed": float("nan")}))
    assert (0, 0) in arch.cells


def test_reinserting_same_genome_id_does_not_duplicate():
    arch = QDArchive()
    b = {"trend_reversal": 0.0, "signal_speed": 0.7}
    arch.insert(_eg("dup", b))
    arch.insert(_eg("dup", b))
    assert sum(len(c) for c in arch.cells.values()) == 1


# ── sampling ──────────────────────────────────────────────────────────────────

def _fill(a: QDArchive) -> QDArchive:
    for i, (tr, sp) in enumerate([(-0.5, 0.2), (0.5, 1.5), (0.0, 0.7)]):
        a.insert(_eg(f"g{i}", {"trend_reversal": tr, "signal_speed": sp}))
    return a


def test_sample_parent_is_deterministic_under_seed():
    s1 = [_fill(QDArchive()).sample_parent(np.random.default_rng(0)).genome.genome_id
          for _ in range(1)]
    a1, a2 = _fill(QDArchive()), _fill(QDArchive())
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    seq1 = [a1.sample_parent(r1).genome.genome_id for _ in range(12)]
    seq2 = [a2.sample_parent(r2).genome.genome_id for _ in range(12)]
    assert seq1 == seq2
    assert QDArchive().sample_parent(np.random.default_rng(0)) is None  # empty → None
    assert s1  # smoke


def test_elites_is_union_of_cells():
    arch = _fill(QDArchive())
    assert {eg.genome.genome_id for eg in arch.elites()} == {"g0", "g1", "g2"}
    assert arch.n_cells() == 3


def test_p7_reuse_penalty_biases_away_from_reused_elite():
    arch = QDArchive(QDConfig(reuse_omega=0.9))
    b = {"trend_reversal": 0.0, "signal_speed": 0.7}   # both in one cell
    arch.insert(_eg("fresh", b, mv=0.1))
    arch.insert(_eg("stale", b, mv=0.2))
    rng = np.random.default_rng(0)
    picks = [arch.sample_parent(rng, reuse_counts={"stale": 50}).genome.genome_id
             for _ in range(40)]
    assert picks.count("fresh") > picks.count("stale")


# ── persistence (frozen bin edges, cells survive resume) ──────────────────────

def test_qd_archive_roundtrip_preserves_cells_and_frozen_edges():
    arch = _fill(QDArchive(QDConfig(grid_dims=2, cell_capacity=2)))
    arch2 = QDArchive.from_dict(arch.to_dict())
    assert arch2.occupancy() == arch.occupancy()
    assert arch2.bin_edges == arch.bin_edges       # edges frozen, identical on resume
    assert {eg.genome.genome_id for eg in arch2.elites()} == {"g0", "g1", "g2"}


# ── controller QD-mode wiring ─────────────────────────────────────────────────

def test_controller_qd_mode_grid_selection_and_book():
    ctrl = EvolutionController(ControllerConfig(selection="qd", seed=0))
    for i, (tr, sp) in enumerate([(-0.5, 0.2), (0.5, 1.5), (0.0, 0.7)]):
        ctrl.insert(_eg(f"g{i}", {"trend_reversal": tr, "signal_speed": sp}))
    assert ctrl.qd is not None and ctrl.qd.n_cells() == 3
    parents = ctrl.select_parents(4)
    assert len(parents) == 4
    assert sum(ctrl.parent_reuse.values()) == 4        # P7 reuse counted
    # accepted book (marginal-value ref + persist source) = union of cell elites
    assert {fid for fid, _ in ctrl.archive_programs()} == {"g0", "g1", "g2"}


def test_controller_qd_state_roundtrip(tmp_path):
    ctrl = EvolutionController(ControllerConfig(selection="qd", seed=1))
    ctrl.insert(_eg("a", {"trend_reversal": -0.5, "signal_speed": 0.2}))
    ctrl.select_parents(2)
    path = tmp_path / "state.json"
    ctrl.save(path)
    ctrl2 = EvolutionController.load(path)
    assert ctrl2.config.selection == "qd"
    assert ctrl2.qd is not None
    assert ctrl2.qd.occupancy() == ctrl.qd.occupancy()
    assert ctrl2.parent_reuse == ctrl.parent_reuse


def test_nsga2_mode_leaves_qd_none_and_archive_unchanged():
    ctrl = EvolutionController(ControllerConfig(selection="nsga2"))
    ctrl.insert(_eg("a", {"trend_reversal": 0.0, "signal_speed": 0.5}))
    assert ctrl.qd is None
    assert {fid for fid, _ in ctrl.archive_programs()} == {"a"}  # from the Pareto archive
