"""Tests for the evolution genome + controller (P1): NSGA-II, archive, N_trials."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund_agent.agents.factor_research.evolution.controller import (
    ControllerConfig,
    EvaluatedGenome,
    EvolutionController,
    constrained_dominates,
    crowding_distance,
    non_dominated_sort,
)
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram, Genome
from quant_fund_agent.research_eval.fitness import (
    FitnessResult,
    GateResults,
    ObjectiveVector,
)


def _fit(cid="c", mv=0.0, ind=0.0, rob=0.0, par=0.0, passed=True,
         n_fail=0) -> FitnessResult:
    # ``rob`` is accepted for call-site compatibility but no longer a Pareto axis.
    gates = GateResults(coverage_ok=True, deflation_ok=True, cost_ok=True)
    if not passed:
        fails = ["coverage_ok", "deflation_ok", "cost_ok"][:max(1, n_fail)]
        for g in fails:
            setattr(gates, g, False)
    return FitnessResult(
        candidate_id=cid,
        objective=ObjectiveVector(marginal_value=mv, independence=ind,
                                  parsimony=par),
        gates=gates,
        diagnostics={"n_trials": 1},
    )


def _genome(gid="g1", fid=None, code=None, island=0, gen=0) -> Genome:
    fid = fid or gid
    code = code or f"def calc():\n    return {gid!r}\n"
    return Genome(genome_id=gid,
                  programs=[FactorProgram(factor_id=fid, code=code)],
                  generation=gen, island=island)


def _eg(gid, **fit_kw) -> EvaluatedGenome:
    return EvaluatedGenome(genome=_genome(gid), fitness=_fit(cid=gid, **fit_kw))


# ── genome ────────────────────────────────────────────────────────────────────

def test_genome_roundtrip_and_fingerprint():
    g = _genome("gx", fid="fx", code="a = 1\nb = 2\n")
    g2 = Genome.from_dict(g.to_dict())
    assert g2.genome_id == "gx" and g2.program.factor_id == "fx"
    # fingerprint ignores whitespace-only differences
    g3 = _genome("gy", fid="fx", code="a = 1\n\n   b = 2\n")
    assert g.code_fingerprint() == g3.code_fingerprint()
    g4 = _genome("gz", fid="fx", code="a = 1\nb = 3\n")
    assert g.code_fingerprint() != g4.code_fingerprint()


def test_single_genome_must_hold_one_program():
    with pytest.raises(ValueError):
        Genome(genome_id="bad", unit="single",
               programs=[FactorProgram("a", "x"), FactorProgram("b", "y")])
    with pytest.raises(ValueError):
        Genome(genome_id="bad", unit="single", programs=[])


# ── constrained dominance + sorting ──────────────────────────────────────────

def test_gate_passer_dominates_gate_failer_regardless_of_objectives():
    weak_passer = _fit("a", mv=-1, ind=-1, rob=-1, par=-1, passed=True)
    strong_failer = _fit("b", mv=9, ind=9, rob=9, par=9, passed=False)
    assert constrained_dominates(weak_passer, strong_failer)
    assert not constrained_dominates(strong_failer, weak_passer)


def test_fewer_failed_gates_wins_between_failers():
    one_fail = _fit("a", passed=False, n_fail=1)
    two_fail = _fit("b", mv=5, passed=False, n_fail=2)
    assert constrained_dominates(one_fail, two_fail)


def test_non_dominated_sort_fronts():
    # c0 dominates c1; c2 is on the front (trades off axes with c0)
    r = [_fit("c0", mv=2, ind=2, rob=2, par=2),
         _fit("c1", mv=1, ind=1, rob=1, par=1),
         _fit("c2", mv=3, ind=1, rob=2, par=2)]
    fronts = non_dominated_sort(r)
    assert sorted(fronts[0]) == [0, 2]
    assert fronts[1] == [1]


def test_crowding_distance_boundary_is_infinite():
    r = [_fit("a", mv=0, ind=0, rob=0, par=0),
         _fit("b", mv=1, ind=1, rob=1, par=1),
         _fit("c", mv=2, ind=2, rob=2, par=2)]
    d = crowding_distance(r, [0, 1, 2])
    assert d[0] == float("inf") and d[2] == float("inf")
    assert np.isfinite(d[1])


# ── controller behaviour ──────────────────────────────────────────────────────

def test_n_trials_increments_and_duplicates_are_flagged():
    ctrl = EvolutionController(ControllerConfig(population_size=4, seed=1))
    assert ctrl.next_trial() == 1
    assert ctrl.next_trial() == 2
    eg = _eg("g1", mv=1)
    ctrl.insert(eg)
    assert ctrl.is_duplicate(eg.genome)
    assert not ctrl.is_duplicate(_genome("g2", code="other = 1\n"))


def test_archive_only_admits_gate_passers_and_stays_non_dominated():
    ctrl = EvolutionController(ControllerConfig(population_size=8, seed=0))
    ctrl.insert(_eg("fail_hi", mv=99, passed=False))
    assert ctrl.archive == []

    ctrl.insert(_eg("mid", mv=1, ind=1, rob=1, par=1))
    ctrl.insert(_eg("best", mv=2, ind=2, rob=2, par=2))     # dominates mid
    ctrl.insert(_eg("tradeoff", mv=3, ind=0, rob=1, par=1))  # non-dominated
    ids = sorted(eg.genome.genome_id for eg in ctrl.archive)
    assert ids == ["best", "tradeoff"]

    book = dict(ctrl.archive_programs())
    assert set(book) == {"best", "tradeoff"}


def test_population_truncates_by_rank():
    ctrl = EvolutionController(ControllerConfig(population_size=2, seed=0))
    ctrl.insert(_eg("worst", mv=0, ind=0, rob=0, par=0))
    ctrl.insert(_eg("ok", mv=1, ind=1, rob=1, par=1))
    ctrl.insert(_eg("best", mv=2, ind=2, rob=2, par=2))
    pop_ids = {eg.genome.genome_id for eg in ctrl.population(0)}
    assert pop_ids == {"ok", "best"}


def test_select_parents_prefers_the_front():
    ctrl = EvolutionController(ControllerConfig(population_size=8, seed=42))
    ctrl.insert(_eg("weak", mv=0, ind=0, rob=0, par=0))
    ctrl.insert(_eg("strong", mv=5, ind=5, rob=5, par=5))
    picks = [p.genome.genome_id for p in ctrl.select_parents(50)]
    assert picks.count("strong") > picks.count("weak")


def test_islands_and_migration():
    ctrl = EvolutionController(ControllerConfig(population_size=4, n_islands=2,
                                                migration_k=1, seed=0))
    a = EvaluatedGenome(genome=_genome("a", island=0), fitness=_fit("a", mv=5))
    b = EvaluatedGenome(genome=_genome("b", island=1, code="bb = 1\n"),
                        fitness=_fit("b", mv=1))
    ctrl.insert(a)
    ctrl.insert(b)
    moved = ctrl.migrate()
    assert moved == 2  # ring: a → island 1, b → island 0
    assert any(eg.genome.genome_id == "a" for eg in ctrl.population(1))


def test_mechanism_groups_reserve_separate_pareto_archives():
    ctrl = EvolutionController(ControllerConfig(
        population_size=4, n_mechanism_groups=2, demes_per_group=2, seed=0))
    strong = _eg("strong", mv=10, ind=10, par=10)
    weak = _eg("weak", mv=1, ind=1, par=1)
    strong.genome.mechanism_group_id = 0
    weak.genome.mechanism_group_id = 1
    ctrl.insert(strong)
    ctrl.insert(weak)

    # A globally dominated factor survives because competition is per mechanism
    # group; the accepted book is the union of both reserved fronts.
    assert [eg.genome.genome_id for eg in ctrl.group_archive(0)] == ["strong"]
    assert [eg.genome.genome_id for eg in ctrl.group_archive(1)] == ["weak"]
    assert {eg.genome.genome_id for eg in ctrl.accepted_book()} == {"strong", "weak"}


def test_migration_never_crosses_mechanism_group_boundary():
    ctrl = EvolutionController(ControllerConfig(
        population_size=4, n_mechanism_groups=2, demes_per_group=2,
        migration_k=1, seed=0))
    for group in range(2):
        for deme in range(2):
            gid = f"g{group}d{deme}"
            eg = _eg(gid, mv=float(group + deme))
            eg.genome.mechanism_group_id = group
            eg.genome.deme_id = deme
            ctrl.insert(eg)
    ctrl.migrate()

    for flat, population in enumerate(ctrl.islands):
        group, deme = ctrl.coordinates(flat)
        assert population
        assert all(eg.genome.mechanism_group_id == group for eg in population)
        assert all(eg.genome.deme_id == deme for eg in population)


def test_duplicate_scope_is_per_deme():
    ctrl = EvolutionController(ControllerConfig(
        n_mechanism_groups=1, demes_per_group=2, seed=0))
    first = _genome("a", code="same = 1\n")
    first.deme_id = 0
    ctrl.insert(EvaluatedGenome(first, _fit("a")))
    same_other_deme = _genome("b", code="same = 1\n")
    same_other_deme.deme_id = 1
    assert not ctrl.is_duplicate(same_other_deme)


def test_state_roundtrip(tmp_path):
    ctrl = EvolutionController(ControllerConfig(population_size=4, seed=7))
    ctrl.next_trial()
    ctrl.insert(_eg("g1", mv=2, ind=1, rob=1, par=0))
    ctrl.insert(_eg("g2", mv=1, ind=2, rob=1, par=0))
    ctrl.generation = 3
    path = tmp_path / "state.json"
    ctrl.save(path)

    loaded = EvolutionController.load(path)
    assert loaded.n_trials == 1
    assert loaded.generation == 3
    assert {eg.genome.genome_id for eg in loaded.archive} == \
           {eg.genome.genome_id for eg in ctrl.archive}
    assert loaded.is_duplicate(_genome("g1"))
    # loaded fitness objects still sort identically
    assert len(loaded.population()) == 2


def test_kept_pool_accumulates_gate_passers_and_roundtrips(tmp_path):
    ctrl = EvolutionController(ControllerConfig(population_size=8, seed=0))
    ctrl.insert(_eg("a", mv=2, ind=2, rob=2, par=2))            # passes → pool + archive
    ctrl.insert(_eg("b", mv=1, ind=1, rob=1, par=1))            # dominated by a, still gate-passing
    ctrl.insert(_eg("fail", mv=9, passed=False))               # gate-fail → NOT in pool
    # the pool keeps every gate-passer (even the dominated b), unlike the archive
    assert {eg.genome.genome_id for eg in ctrl.kept_pool} == {"a", "b"}
    assert {eg.genome.genome_id for eg in ctrl.archive} == {"a"}
    assert dict(ctrl.kept_pool_programs()).keys() == {"a", "b"}

    path = tmp_path / "state.json"
    ctrl.save(path)
    loaded = EvolutionController.load(path)
    assert {eg.genome.genome_id for eg in loaded.kept_pool} == {"a", "b"}
    # a legacy state file (no kept_pool) falls back to the archive
    assert loaded.kept_pool_programs()
