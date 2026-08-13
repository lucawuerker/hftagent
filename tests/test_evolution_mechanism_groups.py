"""Loop-level tests for the two-level diversity model that replaced the QD grid.

The controller's half of this is covered in ``test_evolution_controller.py``
(reserved per-group archives, migration boundaries, per-deme dedup scope).  What
was untested is the *loop's* half: that the knowledge graph actually resolves the
upper level, that the cross-group synthesis operator only exists when there is
more than one group, that seeds and children land in the right coordinates, and
that progressive reveal still behaves when the archive is partitioned by group.

Everything is in-process (``QF_USE_MCP=0``) on a synthetic panel with a fake LLM.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.loop import (
    EvolutionLoop,
    EvolutionRunConfig,
    resolve_mechanism_groups,
)

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _factor_code(fid: str, body: str, horizon: int = 1) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = {horizon}

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


MOM_BODY = "close.pct_change().fillna(0.0)"
VOL_BODY = "stddev(close.pct_change(), 12).fillna(0.0)"
SMOOTH_BODY = "ts_mean(close.pct_change(), 3).fillna(0.0)"
RANK_BODY = "(ts_rank(close, 8) - 0.5).fillna(0.0)"


# ── upper level: knowledge-graph resolution ───────────────────────────────────

def test_single_group_needs_no_graph():
    """The 1-group default must not require a knowledge graph (the ablation arm)."""
    cfg = EvolutionRunConfig(n_mechanism_groups=1, retrieval="none")
    specs = resolve_mechanism_groups(cfg, ["close"])
    assert specs == [{"mechanism_group_id": 0, "community_id": None,
                      "focus": "", "mechanisms": []}]


def test_multiple_groups_require_graphrag():
    """Groups are *defined* by graph communities, so asking for several without
    the graph is a configuration error, not a silent fallback to one group."""
    cfg = EvolutionRunConfig(n_mechanism_groups=4, retrieval="rag")
    with pytest.raises(ValueError, match="require retrieval='graphrag'"):
        resolve_mechanism_groups(cfg, ["close"])


def test_graph_that_cannot_fill_the_groups_is_an_error(monkeypatch):
    """A thin graph must fail loudly rather than quietly collapse the upper level."""
    import quant_fund_agent.knowledge.graph_query as gq

    monkeypatch.setattr(gq, "mechanism_group_specs",
                        lambda graph, n, fields: [
                            {"mechanism_group_id": 0, "community_id": 1,
                             "focus": "momentum", "mechanisms": ["m"]}])
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.graph_store.KnowledgeGraph.load",
        staticmethod(lambda *a, **k: object()))

    cfg = EvolutionRunConfig(n_mechanism_groups=3, retrieval="graphrag")
    with pytest.raises(ValueError, match="could form only 1 usable mechanism groups"):
        resolve_mechanism_groups(cfg, ["close"])


def test_max_mode_uses_however_many_usable_groups_the_graph_forms(monkeypatch):
    """mechanism_groups_mode='max': the requested count is a hard UPPER limit —
    a graph that forms fewer usable groups shrinks the run instead of aborting,
    and the surviving specs are renumbered contiguously."""
    import quant_fund_agent.knowledge.graph_query as gq

    monkeypatch.setattr(gq, "mechanism_group_specs",
                        lambda graph, n, fields: [
                            {"mechanism_group_id": 0, "community_id": 1,
                             "focus": "momentum", "mechanisms": ["m"]},
                            {"mechanism_group_id": 1, "community_id": 2,
                             "focus": "", "mechanisms": []},   # unusable top-up
                            {"mechanism_group_id": 2, "community_id": 3,
                             "focus": "liquidity", "mechanisms": ["l"]}])
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.graph_store.KnowledgeGraph.load",
        staticmethod(lambda *a, **k: object()))

    cfg = EvolutionRunConfig(n_mechanism_groups=8, retrieval="graphrag",
                             mechanism_groups_mode="max")
    specs = resolve_mechanism_groups(cfg, ["close"])
    assert [s["focus"] for s in specs] == ["momentum", "liquidity"]
    assert [s["mechanism_group_id"] for s in specs] == [0, 1]


def test_max_mode_shrinks_the_controller_to_the_resolved_count(monkeypatch):
    """The loop must size the island structure to the RESOLVED group count,
    not the requested upper limit."""
    import quant_fund_agent.knowledge.graph_query as gq
    from quant_fund_agent.agents.factor_research.evolution.loop import EvolutionLoop

    monkeypatch.setattr(gq, "mechanism_group_specs",
                        lambda graph, n, fields: [
                            {"mechanism_group_id": 0, "community_id": 1,
                             "focus": "momentum", "mechanisms": ["m"]},
                            {"mechanism_group_id": 1, "community_id": 3,
                             "focus": "liquidity", "mechanisms": ["l"]}])
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.graph_store.KnowledgeGraph.load",
        staticmethod(lambda *a, **k: object()))

    cfg = EvolutionRunConfig(n_mechanism_groups=8, retrieval="graphrag",
                             mechanism_groups_mode="max", demes_per_group=2)
    loop = EvolutionLoop(cfg, data_context="ctx", fields=["close"])
    assert loop.cfg.n_mechanism_groups == 2
    assert len(loop.controller.islands) == 2 * 2
    assert [s["focus"] for s in loop.mechanism_groups] == ["momentum", "liquidity"]


def test_max_mode_with_zero_usable_groups_still_fails_loudly(monkeypatch):
    import quant_fund_agent.knowledge.graph_query as gq

    monkeypatch.setattr(gq, "mechanism_group_specs",
                        lambda graph, n, fields: [
                            {"mechanism_group_id": 0, "community_id": None,
                             "focus": "", "mechanisms": []}])
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.graph_store.KnowledgeGraph.load",
        staticmethod(lambda *a, **k: object()))

    cfg = EvolutionRunConfig(n_mechanism_groups=8, retrieval="graphrag",
                             mechanism_groups_mode="max")
    with pytest.raises(ValueError, match="no usable mechanism groups"):
        resolve_mechanism_groups(cfg, ["close"])


def test_legacy_islands_alias_means_demes_per_group():
    """Old configs/checkpoints said ``n_islands``; it must map to the lower level."""
    assert EvolutionRunConfig(n_islands=4).effective_demes_per_group == 4
    # an explicit grouped setting wins over the alias
    assert EvolutionRunConfig(n_islands=4,
                              demes_per_group=2).effective_demes_per_group == 2
    assert EvolutionRunConfig().effective_demes_per_group == 1


# ── loop wiring ───────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_panel():
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}


class _FakeLLM:
    """Emits a structurally distinct valid child on every call.

    The window must differ from every seed's *and* from every other child's: the
    dedup fingerprint is id-masked, so a child whose body matches a seed already
    in its (group, deme) is rejected as a duplicate and never reaches the
    lineage — which would silently hollow out the assertions below.
    """

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(str(prompt))
        window = 20 + self.calls          # never 3 (SMOOTH_BODY) or 8 (RANK_BODY)
        payload = {
            "factor_id": f"evo_child_{self.calls}",
            "name": "Evo child",
            "category": "momentum",
            "trading_idea": "short-horizon momentum persists",
            "description": "smoothed momentum",
            "prediction_horizon": 1,
            "suggested_horizons": [1, 3],
            "expected_sign": 1,
            "code": _factor_code(
                f"evo_child_{self.calls}",
                f"ts_mean(close.pct_change(), {window}).fillna(0.0)"),
        }

        class _Resp:
            content = json.dumps(payload)

        return _Resp()


def _children(loop) -> list[dict]:
    """Non-seed, non-bookkeeping lineage rows — the children actually admitted."""
    return [row for row in loop.controller.lineage
            if not row.get("event") and row.get("operator") != "seed"]


@pytest.fixture()
def grouped_loop(synthetic_panel, tmp_path, monkeypatch):
    """Factory for an EvolutionLoop with a configurable two-level population."""
    monkeypatch.setenv("QF_USE_MCP", "0")
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("group-panel",))
    svc._SIGNAL_CACHE.clear()
    fake = _FakeLLM()
    monkeypatch.setattr(loop_mod, "_get_llm", lambda temperature, role=None: fake)

    def _make(**overrides):
        params = dict(
            generations=2, population_size=6, seed=11, target_horizon=1,
            n_mechanism_groups=2, demes_per_group=2, children_per_deme=1,
            n_tickers=None, out_dir=str(tmp_path / "evolution"),
        )
        params.update(overrides)
        loop = EvolutionLoop(EvolutionRunConfig(**params), data_context="CTX",
                             fields=["open", "high", "low", "close", "volume"])
        loop._load_known_ids = lambda: None
        return loop

    return _make, fake, tmp_path


def _seeds() -> list[FactorProgram]:
    """Four seeds — with 2 groups the loop round-robins them two per group."""
    return [
        FactorProgram(factor_id="seed_mom", code=_factor_code("seed_mom", MOM_BODY),
                      name="seed momentum", trading_idea="momentum",
                      expected_sign=1, prediction_horizon=1),
        FactorProgram(factor_id="seed_vol", code=_factor_code("seed_vol", VOL_BODY),
                      name="seed vol", trading_idea="volatility state",
                      expected_sign=1, prediction_horizon=1),
        FactorProgram(factor_id="seed_smooth",
                      code=_factor_code("seed_smooth", SMOOTH_BODY),
                      name="seed smooth", trading_idea="smoothed momentum",
                      expected_sign=1, prediction_horizon=1),
        FactorProgram(factor_id="seed_rank", code=_factor_code("seed_rank", RANK_BODY),
                      name="seed rank", trading_idea="cross-sectional rank",
                      expected_sign=1, prediction_horizon=1),
    ]


def test_seeds_are_distributed_across_groups_and_demes(grouped_loop):
    make, _, _ = grouped_loop
    loop = make(generations=0)
    loop.run(initial_programs=_seeds())

    ctrl = loop.controller
    assert len(ctrl.islands) == 4          # 2 groups × 2 demes
    placed = [eg for isl in ctrl.islands for eg in isl]
    assert len(placed) == 4
    # both groups populated, and each genome's flat island matches its coordinates
    groups = {eg.genome.mechanism_group_id for eg in placed}
    assert groups == {0, 1}
    for eg in placed:
        assert eg.genome.island == ctrl.flat_island(
            eg.genome.mechanism_group_id, eg.genome.deme_id)


def test_each_group_keeps_its_own_reserved_archive(grouped_loop):
    make, _, _ = grouped_loop
    loop = make()
    loop.run(initial_programs=_seeds())

    ctrl = loop.controller
    assert len(ctrl.group_archives) == 2
    # the accepted book is the union of the reserved archives, and every member
    # sits in the archive of the group it belongs to
    assert ctrl.accepted_book() == ctrl.archive
    for group_id, archive in enumerate(ctrl.group_archives):
        for eg in archive:
            assert eg.genome.mechanism_group_id == group_id
    assert sum(len(a) for a in ctrl.group_archives) == len(ctrl.archive)


def test_children_stay_in_their_parent_group(grouped_loop):
    """Ordinary operators are group-local; only the explicit cross-group synthesis
    operator may mix, and it is disabled here."""
    make, _, _ = grouped_loop
    loop = make(p_cross_group=0.0)
    loop.run(initial_programs=_seeds())

    children = _children(loop)
    assert children, "no children were admitted — the assertions below are vacuous"
    for row in children:
        assert row["operator"] not in ("cross_group", "cross_group_splice")
        assert row["island"] == loop.controller.flat_island(
            row["mechanism_group_id"], row["deme_id"])


def test_cross_group_operator_fires_only_when_there_is_more_than_one_group(
        grouped_loop):
    """A high cross-group weight is honoured with 2 groups and forced to zero with
    1 — the operator is meaningless when there is nothing to cross."""
    make, _, _ = grouped_loop

    many = make(p_cross_group=0.9)
    many.run(initial_programs=_seeds())
    assert "cross_group" in {r["operator"] for r in _children(many)}, (
        "p_cross_group=0.9 with 2 groups should produce cross-group children")

    one = make(n_mechanism_groups=1, demes_per_group=2, p_cross_group=0.9)
    one.run(initial_programs=_seeds())
    one_children = _children(one)
    assert one_children, "no children were admitted — the assertion below is vacuous"
    assert "cross_group" not in {r["operator"] for r in one_children}, (
        "cross-group synthesis must be zero-weighted with one group, "
        "not merely unlikely")


def test_group_focus_reaches_the_mutation_prompt(grouped_loop):
    make, fake, _ = grouped_loop
    # Force every child through the LLM operator: which op a child draws is an RNG
    # outcome, so a mixed operator mix can leave the LLM uncalled and the assertion
    # below vacuous.
    loop = make(generations=1, p_llm_semantic=1.0, p_crossover=0.0,
                p_jitter=0.0, p_cross_group=0.0)
    loop.mechanism_groups = [
        {"mechanism_group_id": 0, "community_id": 7,
         "focus": "LIQUIDITY-PROVISION-MECHANISM", "mechanisms": ["liq"]},
        {"mechanism_group_id": 1, "community_id": 9,
         "focus": "SLOW-INFORMATION-DIFFUSION", "mechanisms": ["diff"]},
    ]
    loop.run(initial_programs=_seeds())

    assert fake.prompts, "the LLM operator never ran — nothing to assert on"
    joined = "\n".join(fake.prompts)
    # each group's community brief must reach the children evolving inside it
    assert "LIQUIDITY-PROVISION-MECHANISM" in joined
    assert "SLOW-INFORMATION-DIFFUSION" in joined


def test_mechanism_groups_are_checkpointed(grouped_loop):
    make, _, tmp_path = grouped_loop
    loop = make(generations=0)
    loop.mechanism_groups = [
        {"mechanism_group_id": 0, "community_id": 3, "focus": "A", "mechanisms": []},
        {"mechanism_group_id": 1, "community_id": 4, "focus": "B", "mechanisms": []},
    ]
    loop.run(initial_programs=_seeds())

    stored = json.loads(
        (tmp_path / "evolution" / "mechanism_groups.json").read_text())
    assert [s["focus"] for s in stored] == ["A", "B"]


# ── progressive reveal × mechanism groups ─────────────────────────────────────

def test_progressive_reveal_rescores_every_group_archive(grouped_loop):
    """The reveal path re-prunes per group; a group must not be able to dominate
    another group's members out of existence when the frontier advances."""
    make, _, tmp_path = grouped_loop
    loop = make(progressive=True, test_frac=0.2, seed_frac=0.5,
                reveal_every=1, val_blocks=1)
    loop.run(initial_programs=_seeds())

    ctrl = loop.controller
    # the frontier advanced and each reveal logged an honest prequential score
    prequential = tmp_path / "evolution" / "prequential.jsonl"
    assert prequential.exists()
    rows = [json.loads(line) for line in prequential.read_text().splitlines()]
    assert [r["generation"] for r in rows] == [1, 2]

    # re-scores happened without billing the multiple-testing budget …
    lineage = ctrl.lineage
    assert [r for r in lineage if r.get("event") == "rescore"]
    assert ctrl.n_trials == len([r for r in lineage if "event" not in r])

    # … and the archive is still correctly partitioned by group afterwards
    for group_id, archive in enumerate(ctrl.group_archives):
        for eg in archive:
            assert eg.genome.mechanism_group_id == group_id
            assert eg.fitness.diagnostics.get("scored_through") is not None


# ── graph read-only mode (--graph-readonly, 2026-08-01) ───────────────────────

class _SpyGraph:
    """Fails the test if any mutating call reaches it in read-only mode."""

    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


def test_graph_readonly_is_a_strict_noop():
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        link_programs_into_graph,
    )
    from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram

    graph = _SpyGraph()
    prog = FactorProgram(factor_id="f1", code="def calc(data):\n    return 0\n")
    # readonly: nothing is linked, nothing is saved (no attribute of the spy is
    # touched beyond existence — link/refresh would raise AttributeError on it)
    link_programs_into_graph(graph, [prog], {"f1": "some_mechanism"}, readonly=True)
    assert graph.saved is False


def test_graph_linkback_still_saves_when_writable(tmp_path, monkeypatch):
    from quant_fund_agent.agents.factor_research.evolution import loop as loop_mod

    calls = {"link": 0, "save": 0}

    class _G:
        def save(self):
            calls["save"] += 1

    monkeypatch.setattr(
        "quant_fund_agent.knowledge.empirical_edges.link_factor_to_mechanism",
        lambda graph, fid, mech: calls.__setitem__("link", calls["link"] + 1))
    monkeypatch.setattr(
        "quant_fund_agent.knowledge.empirical_edges.refresh_field_usage",
        lambda graph, inputs: None)
    from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
    prog = FactorProgram(factor_id="f1", code="def calc(data):\n    return 0\n")
    loop_mod.link_programs_into_graph(_G(), [prog], {"f1": "m"}, readonly=False)
    assert calls == {"link": 1, "save": 1}


def test_children_per_deme_zero_is_selection_only(grouped_loop):
    """children_per_deme=0 (the L1H ablation): generation 0 seeds, later
    generations propose NO children — only the deterministic machinery runs."""
    make, fake, _ = grouped_loop
    loop = make(children_per_deme=0, generations=2)
    loop.run(initial_programs=_seeds())

    ops = {row.get("operator") for row in loop.controller.lineage
           if row.get("event") is None}
    assert ops == {"seed"}, f"non-seed children proposed: {ops - {'seed'}}"
    assert loop.controller.generation == 2          # generations still advanced
    assert loop.controller.archive                  # seeds survived selection
