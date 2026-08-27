"""Tests for the non-evolutionary ``--variant refine`` ablation.

Refine mode replaces the evolutionary operators with: per-lineage LLM
refinement of the SAME factor (at most ``refine_rounds`` times), continuous
fresh re-seeding when a deme runs out of refinement work, and the occasional
explicit cross-group synthesis as the only combination operator.  Everything
runs in-process against the synthetic panel, like ``test_evolution_loop``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.factor_research.evolution.controller import (
    EvolutionController,
)
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.loop import (
    EvolutionLoop,
    EvolutionRunConfig,
)

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _factor_code(fid: str, body: str, inputs=("close",), horizon: int = 1) -> str:
    inputs_lit = ", ".join(f'"{f}"' for f in inputs)
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, stddev, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = [{inputs_lit}]
    prediction_horizon = {horizon}

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


MOM_BODY = 'close.pct_change().fillna(0.0)'
VOL_BODY = 'stddev(close.pct_change(), 12).fillna(0.0)'


@pytest.fixture()
def synthetic_panel():
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                         columns=TICKERS)
    return {"close": close}


class FakeLLM:
    """Emits a structurally UNIQUE child per call (fresh window constant), so
    the code-fingerprint dedup never silently swallows a refinement."""

    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.calls += 1
        self.prompts.append(str(prompt))
        body = f'ts_mean(close.pct_change(), {2 + self.calls}).fillna(0.0)'
        payload = {
            "factor_id": f"refined_child_{self.calls}",
            "name": "Refined child",
            "category": "momentum",
            "trading_idea": "short-horizon momentum persists",
            "description": "smoothed momentum",
            "prediction_horizon": 1,
            "suggested_horizons": [1],
            "expected_sign": 1,
            "code": _factor_code(f"refined_child_{self.calls}", body),
        }

        class _Resp:
            content = json.dumps(payload)

        return _Resp()


def _wire(synthetic_panel, tmp_path, monkeypatch, **cfg_kwargs):
    monkeypatch.setenv("QF_USE_MCP", "0")

    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("test-panel",))
    svc._SIGNAL_CACHE.clear()

    fake = FakeLLM()
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    monkeypatch.setattr(loop_mod, "_get_llm", lambda temperature, role=None: fake)

    cfg_kwargs.setdefault("p_cross_group", 0.0)
    cfg = EvolutionRunConfig(
        variant="refine", population_size=8, children_per_deme=1,
        seed=11, target_horizon=1, stability_blocks=4,
        n_tickers=None, out_dir=str(tmp_path / "evolution"),
        **cfg_kwargs,
    )
    loop = EvolutionLoop(cfg, data_context="TEST DATA CONTEXT",
                         fields=["open", "high", "low", "close", "volume"])
    loop._load_known_ids = lambda: None
    return loop, fake


def _seeds() -> list[FactorProgram]:
    return [
        FactorProgram(factor_id="seed_mom", code=_factor_code("seed_mom", MOM_BODY),
                      name="seed momentum", trading_idea="momentum",
                      expected_sign=1, prediction_horizon=1),
        FactorProgram(factor_id="seed_vol", code=_factor_code("seed_vol", VOL_BODY),
                      name="seed vol", trading_idea="volatility state",
                      expected_sign=1, prediction_horizon=1),
    ]


def _rows(loop) -> list[dict]:
    return [r for r in loop.controller.lineage if "operator" in r]


def test_refine_chains_same_factor_and_respects_cap(
    synthetic_panel, tmp_path, monkeypatch
):
    loop, fake = _wire(synthetic_panel, tmp_path, monkeypatch,
                       generations=5, refine_rounds=2)
    # No fresh seeding in this test: exhausted demes simply idle.
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    monkeypatch.setattr(loop_mod, "seed_programs",
                        lambda *a, **k: [])
    loop.run(initial_programs=_seeds())

    rows = _rows(loop)
    ops = {r["operator"] for r in rows}
    assert ops <= {"seed", "refine"}, ops
    refines = [r for r in rows if r["operator"] == "refine"]
    # 2 lineages x refine_rounds=2 = 4 refinements, one slot per generation.
    assert len(refines) == 4
    assert all(len(r["parent_ids"]) == 1 for r in refines)

    # Every refinement chains off the lineage's LATEST version, and no root
    # lineage is refined more than refine_rounds times.
    parent_of = {r["genome_id"]: r["parent_ids"][0] for r in refines}
    seeds = {r["genome_id"] for r in rows if r["operator"] == "seed"}

    def _root(gid: str) -> str:
        while gid in parent_of:
            gid = parent_of[gid]
        return gid

    depth: dict[str, int] = {}
    for r in refines:
        root = _root(r["genome_id"])
        assert root in seeds
        depth[root] = depth.get(root, 0) + 1
    assert all(d <= 2 for d in depth.values())
    # Each refinement's parent is either the seed or the previous refinement of
    # the same lineage — never a sibling from another lineage (no crossover).
    for r in refines:
        parent = r["parent_ids"][0]
        assert parent in seeds or parent in parent_of

    # The refine prompt is the same-factor prompt, not the mutation prompt.
    assert any("REFINED VERSION of THIS SAME factor" in p for p in fake.prompts)
    assert not any("evolutionary search" in p and "improved CHILD factor" in p
                   for p in fake.prompts)


def test_fresh_seeding_fills_exhausted_demes(synthetic_panel, tmp_path, monkeypatch):
    loop, fake = _wire(synthetic_panel, tmp_path, monkeypatch,
                       generations=4, refine_rounds=1)

    counter = {"n": 0}

    def _fake_seed_programs(cfg, data_context, existing_ids, fields=None,
                            mechanism_group=None, prompt_ids=None):
        counter["n"] += 1
        fid = f"fresh_seed_{counter['n']}"
        body = f'ts_rank(close, {5 + counter["n"]}).fillna(0.0)'
        return [FactorProgram(factor_id=fid, code=_factor_code(fid, body),
                              name=fid, trading_idea="fresh idea",
                              expected_sign=1, prediction_horizon=1)]

    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    monkeypatch.setattr(loop_mod, "seed_programs", _fake_seed_programs)
    loop.run(initial_programs=_seeds()[:1])

    rows = _rows(loop)
    mid_run_seeds = [r for r in rows
                     if r["operator"] == "seed" and r["generation"] > 0]
    assert mid_run_seeds, "exhausted deme never re-seeded"
    # A freshly seeded lineage is itself refined in a later generation.
    fresh_ids = {r["genome_id"] for r in mid_run_seeds}
    assert any(r["parent_ids"][0] in fresh_ids
               for r in rows if r["operator"] == "refine")


def test_cross_group_is_only_combination_operator(
    synthetic_panel, tmp_path, monkeypatch
):
    loop, fake = _wire(synthetic_panel, tmp_path, monkeypatch,
                       generations=2, refine_rounds=2,
                       n_mechanism_groups=2, p_cross_group=1.0)
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    monkeypatch.setattr(loop_mod, "seed_programs", lambda *a, **k: [])
    seeds = _seeds()
    seeds[0].mechanism_group_id, seeds[0].mechanism = 0, "mech_a"
    seeds[1].mechanism_group_id, seeds[1].mechanism = 1, "mech_b"
    loop.run(initial_programs=seeds)

    rows = _rows(loop)
    ops = {r["operator"] for r in rows}
    assert "cross_group" in ops
    assert ops <= {"seed", "refine", "cross_group"}
    for r in rows:
        if r["operator"] == "cross_group":
            assert len(r["parent_ids"]) == 2


def test_refine_state_checkpoints_and_resumes(
    synthetic_panel, tmp_path, monkeypatch
):
    loop, fake = _wire(synthetic_panel, tmp_path, monkeypatch,
                       generations=2, refine_rounds=3)
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    monkeypatch.setattr(loop_mod, "seed_programs", lambda *a, **k: [])
    loop.run(initial_programs=_seeds())

    state_path = tmp_path / "evolution" / "refine_state.json"
    assert state_path.exists()
    saved = json.loads(state_path.read_text())
    saved_entries = [e for entries in saved.values() for e in entries]
    assert saved_entries
    assert all(0 <= e["refines"] <= 3 for e in saved_entries)

    # Fresh process: reload controller + refine state, continue to the end.
    cfg2 = EvolutionRunConfig(**{**loop.cfg.to_dict(),
                                 "plateau_scales": tuple(loop.cfg.plateau_scales),
                                 "generations": 4})
    loop2 = EvolutionLoop(cfg2, data_context="TEST DATA CONTEXT",
                          fields=["open", "high", "low", "close", "volume"])
    loop2._load_known_ids = lambda: None
    loop2.controller = EvolutionController.load(
        tmp_path / "evolution" / "state.json")
    loop2.run(resume=True)

    restored = {e["eg"].genome.genome_id: e["refines"]
                for entries in loop2._refine_entries.values()
                for e in entries}
    by_id = {e["genome_id"]: e["refines"] for e in saved_entries}
    # Whatever is still pending after the resumed run must have refine counts
    # >= the checkpointed ones (the resumed generations only add refinements).
    for gid, refines in restored.items():
        if gid in by_id:
            assert refines >= by_id[gid]
    rows = _rows(loop2)
    assert {r["operator"] for r in rows} <= {"seed", "refine"}
    resumed_refines = [r for r in rows
                       if r["operator"] == "refine" and r["generation"] > 2]
    assert resumed_refines, "resumed run proposed no refinements"


def test_mid_seeding_kill_resumes_without_reseeding_done_groups(
    synthetic_panel, tmp_path, monkeypatch
):
    """A run killed DURING generation-0 seeding checkpoints after every group;
    the restart must skip the already-seeded groups (no re-spend) and finish."""
    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod

    seeded_groups: list[int] = []

    def _seed_programs(cfg, data_context, existing_ids, fields=None,
                       mechanism_group=None, *, die_on=None, prompt_ids=None):
        gid = int((mechanism_group or {}).get("mechanism_group_id", 0))
        if die_on is not None and gid == die_on:
            raise KeyboardInterrupt("simulated 30-min kill")
        seeded_groups.append(gid)
        fid = f"g{gid}_seed"
        # windows far from the FakeLLM children's (2+calls) so the canonical
        # code-fingerprint dedup never collides with a refinement child
        body = f'ts_mean(close.pct_change(), {30 + gid}).fillna(0.0)'
        return [FactorProgram(factor_id=fid, code=_factor_code(fid, body),
                              name=fid, trading_idea="seed", expected_sign=1,
                              prediction_horizon=1,
                              mechanism_group_id=gid, mechanism=f"m{gid}")]

    specs = [{"mechanism_group_id": 0, "community_id": None,
              "focus": "focus 0", "mechanisms": []},
             {"mechanism_group_id": 1, "community_id": None,
              "focus": "focus 1", "mechanisms": []}]
    monkeypatch.setattr(loop_mod, "resolve_mechanism_groups",
                        lambda cfg, fields: specs)
    loop, fake = _wire(synthetic_panel, tmp_path, monkeypatch,
                       generations=1, refine_rounds=1, n_mechanism_groups=2)
    monkeypatch.setattr(
        loop_mod, "seed_programs",
        lambda *a, **k: _seed_programs(*a, **k, die_on=1))
    with pytest.raises(KeyboardInterrupt):
        loop.run()
    assert seeded_groups == [0]
    state_path = tmp_path / "evolution" / "state.json"
    assert state_path.exists()

    # Restart: group 0 must NOT be re-seeded; group 1 is seeded and the run
    # completes, with group 0's lineage restored for refinement.
    loop2, _ = _wire(synthetic_panel, tmp_path, monkeypatch,
                     generations=1, refine_rounds=1, n_mechanism_groups=2)
    monkeypatch.setattr(loop_mod, "seed_programs", _seed_programs)
    loop2.controller = EvolutionController.load(state_path)
    loop2.run(resume=True)
    assert seeded_groups == [0, 1], "restart re-seeded an already-done group"
    rows = _rows(loop2)
    seed_fids = [fid for r in rows if r["operator"] == "seed"
                 for fid in r["factor_ids"] if r["generation"] == 0]
    assert sorted(seed_fids) == ["g0_seed", "g1_seed"]
    refined_parents = {r["parent_ids"][0] for r in rows
                       if r["operator"] == "refine"}
    seed_gids = {r["genome_id"] for r in rows if r["operator"] == "seed"
                 and r["generation"] == 0}
    assert refined_parents and refined_parents <= seed_gids


def test_refine_variant_rejects_set_unit():
    with pytest.raises(ValueError):
        EvolutionLoop(EvolutionRunConfig(variant="refine", unit="set"),
                      data_context="X", fields=["close"])
