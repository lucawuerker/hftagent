"""End-to-end test of the P1 evolutionary loop on a synthetic panel.

Everything runs in-process (no MCP subprocess, no network): the panel loader is
monkeypatched to a synthetic momentum-friendly panel, and the LLM is a fake that
emits deterministic factor programs.  This exercises the full path:
seed → evaluate (research_eval harness) → NSGA-II insert → mutate (LLM + jitter)
→ dedup → archive → checkpoint → persist.
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
    persist_archive,
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
CHILD_BODY_1 = 'ts_mean(close.pct_change(), 3).fillna(0.0)'
CHILD_BODY_2 = '(ts_rank(close, 8) - 0.5).fillna(0.0)'
TEST_BRANCH_BODY = (
    '(1.0 if float(close.iloc[-1, 0]) < 1_000_000.0 else -1.0) '
    '* close.pct_change().fillna(0.0)'
)


@pytest.fixture()
def synthetic_panel():
    """AR(1)-momentum returns so a 1-bar momentum factor has genuine, robust edge."""
    rng = np.random.default_rng(3)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}


class FakeLLM:
    """Deterministic stand-in for the mutating LLM: cycles child programs."""

    def __init__(self):
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        body = CHILD_BODY_1 if self.calls % 2 else CHILD_BODY_2
        payload = {
            "factor_id": "evo_child",
            "name": "Evo child",
            "category": "momentum",
            "trading_idea": "short-horizon momentum persists",
            "description": "smoothed momentum",
            "prediction_horizon": 1,
            "suggested_horizons": [1, 3],
            "expected_sign": 1,
            "code": _factor_code("evo_child", body),
        }

        class _Resp:
            content = json.dumps(payload)

        return _Resp()


@pytest.fixture()
def wired_loop(synthetic_panel, tmp_path, monkeypatch):
    """An EvolutionLoop wired to the synthetic panel + fake LLM, in-process."""
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

    cfg = EvolutionRunConfig(
        generations=2, population_size=6, children_per_generation=4,
        seed=11, target_horizon=1, cpcv_groups=4, cpcv_k=1,
        n_tickers=None, out_dir=str(tmp_path / "evolution"),
    )
    loop = EvolutionLoop(cfg, data_context="TEST DATA CONTEXT",
                         fields=["open", "high", "low", "close", "volume"])
    loop._load_known_ids = lambda: None  # keep the test hermetic
    return loop, fake, tmp_path


def _seeds() -> list[FactorProgram]:
    return [
        FactorProgram(factor_id="seed_mom", code=_factor_code("seed_mom", MOM_BODY),
                      name="seed momentum", trading_idea="momentum", expected_sign=1,
                      prediction_horizon=1),
        FactorProgram(factor_id="seed_vol", code=_factor_code("seed_vol", VOL_BODY),
                      name="seed vol", trading_idea="volatility state", expected_sign=1,
                      prediction_horizon=1),
    ]


def test_evaluate_fitness_does_not_expose_test_rows_to_factor_calc(
    synthetic_panel, monkeypatch
):
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.splits import three_way_split

    split = three_way_split(synthetic_panel["close"].index, is_frac=0.5, val_frac=0.25)
    poisoned_panel = {"close": synthetic_panel["close"].copy()}
    poisoned_panel["close"].iloc[split.test_mask] = 2_000_000.0

    candidate = {
        "factor_id": "test_branch_probe",
        "code": _factor_code("test_branch_probe", TEST_BRANCH_BODY),
        "expected_sign": 1,
    }
    kwargs = dict(
        target_horizon=1,
        is_frac=0.5,
        val_frac=0.25,
        cpcv_groups=4,
        cpcv_k=1,
        fields=["close"],
        n_tickers=None,
    )

    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("test-branch-panel",))
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    svc._SIGNAL_CACHE.clear()
    clean = svc.evaluate_fitness(candidate, **kwargs)

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: poisoned_panel)
    svc._SIGNAL_CACHE.clear()
    poisoned = svc.evaluate_fitness(candidate, **kwargs)

    assert clean["ok"], clean.get("error")
    assert poisoned["ok"], poisoned.get("error")
    assert clean["fitness"] == poisoned["fitness"]
    assert clean["fitness"]["raw"]["split_sizes"] == {"is": 240, "val": 120, "test": 0}
    assert clean["fitness"]["raw"]["heldout_test_split_sizes"] == {
        "is": 240,
        "val": 120,
        "test": 120,
    }


def test_full_loop_runs_and_checkpoints(wired_loop):
    loop, fake, tmp_path = wired_loop
    summary = loop.run(initial_programs=_seeds())

    # every scored candidate was billed against the deflation budget
    assert summary["n_trials"] >= 2                       # both seeds scored
    assert summary["generations"] == 2
    assert fake.calls > 0                                 # the LLM actually mutated

    # the momentum seed has real edge on the AR(1) panel → it must be selectable
    assert summary["archive"], "archive should not be empty on a momentum panel"
    archive_ids = {fid for a in summary["archive"] for fid in a["factor_ids"]}
    assert "seed_mom" in archive_ids

    # checkpoints exist and reload cleanly
    state_path = tmp_path / "evolution" / "state.json"
    assert state_path.exists()
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvolutionController,
    )

    reloaded = EvolutionController.load(state_path)
    assert reloaded.n_trials == summary["n_trials"]
    assert {eg.genome.program.factor_id for eg in reloaded.archive} == archive_ids

    lineage = [json.loads(l) for l in
               (tmp_path / "evolution" / "lineage.jsonl").read_text().splitlines()]
    assert len(lineage) == summary["n_trials"]            # one row per scored candidate
    assert {row["operator"] for row in lineage} & {"llm_semantic", "crossover", "jitter"}
    # children carry their parents in the lineage
    child_rows = [r for r in lineage if r["generation"] > 0]
    assert child_rows and all(r["parent_ids"] for r in child_rows
                              if r["operator"] != "seed")


def test_duplicate_children_are_not_rebilled(wired_loop):
    loop, fake, _ = wired_loop
    loop.run(initial_programs=_seeds())
    # FakeLLM only knows two distinct child programs (plus jitters); with 8
    # child slots at least one duplicate must have been proposed and skipped.
    distinct_codes = {eg.genome.code_fingerprint()
                      for eg in loop.controller.population()}
    assert loop.controller.n_trials <= 2 + 8              # never more than proposals
    assert len(distinct_codes) == len(
        {eg.genome.genome_id for eg in loop.controller.population()})


def test_fixed_book_is_passed_to_single_candidate_evaluator(tmp_path, monkeypatch):
    from quant_fund_agent.mcp import research_client
    from quant_fund_agent.research_eval.fitness import (
        FitnessResult,
        GateResults,
        ObjectiveVector,
    )

    fixed = {"factor_id": "fixed_mom", "code": _factor_code("fixed_mom", MOM_BODY)}
    cfg = EvolutionRunConfig(
        target_horizon=1,
        fixed_book=[fixed],
        out_dir=str(tmp_path / "evolution"),
    )
    loop = EvolutionLoop(cfg, data_context="CTX", fields=["close"])
    captured: dict = {}

    def fake_evaluate_fitness(candidate, book, jitter, **kwargs):
        captured["candidate"] = candidate
        captured["book"] = book
        captured["kwargs"] = kwargs
        fitness = FitnessResult(
            candidate_id=candidate["factor_id"],
            objective=ObjectiveVector(marginal_value=0.1, independence=0.2),
            gates=GateResults(coverage_ok=True, degradation_ok=None,
                              deflation_ok=True),
            diagnostics={"n_trials": kwargs["n_trials"]},
        )
        return {"ok": True, "fitness": fitness.to_dict()}

    monkeypatch.setattr(research_client, "evaluate_fitness", fake_evaluate_fitness)

    result = loop.evaluate_program(
        FactorProgram(factor_id="candidate_mom",
                      code=_factor_code("candidate_mom", MOM_BODY))
    )

    assert result is not None
    assert captured["candidate"]["factor_id"] == "candidate_mom"
    assert captured["book"] == [{"factor_id": "fixed_mom", "code": fixed["code"]}]
    assert loop.fixed_book[0]["factor_id"] == "fixed_mom"


def test_persist_archive_uses_the_oneshot_path(wired_loop, monkeypatch):
    loop, _, _ = wired_loop
    loop.run(initial_programs=_seeds())
    assert loop.controller.archive

    from quant_fund_agent.mcp import research_client

    materialised, persisted = [], {}

    def fake_materialise(fid, code):
        materialised.append(fid)
        return {"ok": True, "code_path": f"/tmp/fake/{fid}.py"}

    def fake_backtest(factor_ids, **kw):
        return {fid: {"ok": False, "error": "skipped in test"} for fid in factor_ids}

    def fake_persist(kept, rejected):
        persisted["kept"] = kept
        return {"kept_factor_ids": [r["id"] for r in kept], "rejected_factor_ids": []}

    monkeypatch.setattr(research_client, "materialise_factor", fake_materialise)
    monkeypatch.setattr(research_client, "backtest_factors", fake_backtest)
    monkeypatch.setattr(research_client, "persist_results", fake_persist)

    out = persist_archive(loop.controller, session_id="evolution:test:evo",
                          target_horizon=1)
    assert set(out["kept_factor_ids"]) == {
        p.factor_id for eg in loop.controller.archive for p in eg.genome.programs}
    rec = persisted["kept"][0]
    assert rec["source"] == "researcher"
    assert rec["metadata"]["engine"] == "evolution"
    evo = rec["metadata"]["evolution"]
    assert {"genome_id", "generation", "operator", "objective", "gates"} <= set(evo)


def test_two_stage_curation_persists_the_curated_pool(wired_loop, monkeypatch):
    loop, _, _ = wired_loop
    loop.run(initial_programs=_seeds())
    assert loop.controller.kept_pool, "gate-passers should accumulate in the pool"
    pool_ids = {fid for fid, _ in loop.controller.kept_pool_programs()}

    from quant_fund_agent.mcp import research_client

    materialised: list[str] = []
    persisted: dict = {}
    monkeypatch.setattr(research_client, "materialise_factor",
                        lambda fid, code: (materialised.append(fid)
                                           or {"ok": True, "code_path": f"/tmp/{fid}.py"}))
    monkeypatch.setattr(research_client, "backtest_factors",
                        lambda factor_ids, **kw: {f: {"ok": False} for f in factor_ids})
    monkeypatch.setattr(research_client, "persist_results",
                        lambda kept, rej: (persisted.update(kept=kept)
                                           or {"kept_factor_ids": [r["id"] for r in kept],
                                               "rejected_factor_ids": []}))

    # curate the pool (greedy) rather than persisting the Pareto archive
    out = persist_archive(loop.controller, session_id="evolution:test:evo",
                          target_horizon=1, curation="greedy",
                          is_frac=0.6, val_frac=0.2,
                          fields=["open", "high", "low", "close", "volume"])
    kept = set(out["kept_factor_ids"])
    assert kept and kept <= pool_ids                       # curated FROM the whole pool
    assert set(materialised) == kept                       # only curated ids materialised
    assert persisted["kept"][0]["metadata"]["evolution"]["curation"] == "greedy"


def test_eval_failure_is_not_billed(wired_loop):
    loop, _, _ = wired_loop
    broken = FactorProgram(
        factor_id="needs_missing_field",
        code=_factor_code("needs_missing_field",
                          'data["trade"].fillna(0.0)', inputs=("close", "trade")),
    )
    before = loop.controller.n_trials
    assert loop.evaluate_program(broken) is None
    assert loop.controller.n_trials == before
    assert loop.failures and "trade" in loop.failures[0]["error"]


def test_experience_memory_is_written_across_a_run(synthetic_panel, tmp_path,
                                                   monkeypatch):
    """WS5: a run with --memory records per-mechanism attempt tallies + survivors to
    a per-config memory graph, so a later run can read them."""
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("test-panel",))
    svc._SIGNAL_CACHE.clear()

    import quant_fund_agent.agents.factor_research.evolution.loop as loop_mod
    from quant_fund_agent.knowledge import experience
    from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

    monkeypatch.setattr(loop_mod, "_get_llm", lambda temperature, role=None: FakeLLM())
    mem_path = tmp_path / "mem" / "graph.json"
    monkeypatch.setattr(experience, "memory_graph_path", lambda cfg: mem_path)

    cfg = EvolutionRunConfig(
        generations=1, population_size=4, children_per_generation=2, seed=5,
        target_horizon=1, cpcv_groups=4, cpcv_k=1, n_tickers=None,
        out_dir=str(tmp_path / "evo"), memory=True, memory_config="testcfg")
    loop = EvolutionLoop(cfg, data_context="CTX",
                         fields=["open", "high", "low", "close", "volume"])
    loop._load_known_ids = lambda: None
    seeds = [FactorProgram(factor_id="s_mom",
                           code=_factor_code("s_mom", MOM_BODY), name="m",
                           category="momentum", expected_sign=1, prediction_horizon=1)]
    loop.run(initial_programs=seeds)

    assert mem_path.exists()                       # writeback happened
    g = KnowledgeGraph.load(mem_path)
    mid = KnowledgeGraph.mechanism_id("momentum")
    assert g.g.nodes[mid]["n_attempts"] >= 1       # attempt tally recorded (topic=category)


def test_resume_continues_generations_and_trials(wired_loop):
    """E1 block API: resume=True reloads state.json and runs MORE generations,
    preserving n_trials, the archive and the lineage file (append, not wipe)."""
    loop, fake, tmp_path = wired_loop
    s1 = loop.run(initial_programs=_seeds())
    assert s1["generations"] == 2
    lineage_before = (tmp_path / "evolution" / "lineage.jsonl").read_text().splitlines()

    resumed = EvolutionLoop(loop.cfg, data_context="TEST DATA CONTEXT",
                            fields=["open", "high", "low", "close", "volume"])
    resumed._load_known_ids = lambda: None
    s2 = resumed.run(resume=True, n_generations=1)

    assert s2["generations"] == 3                       # 2 + 1 more
    assert s2["n_trials"] >= s1["n_trials"]             # counter carried over
    archive_ids = {fid for a in s2["archive"] for fid in a["factor_ids"]}
    assert "seed_mom" in archive_ids                    # archive survived the reload

    lineage_after = (tmp_path / "evolution" / "lineage.jsonl").read_text().splitlines()
    assert len(lineage_after) >= len(lineage_before)    # old rows kept
    assert lineage_after[: len(lineage_before)] == lineage_before
    # gen-0 seeds were NOT re-admitted on resume
    seed_rows = [json.loads(l) for l in lineage_after
                 if json.loads(l)["operator"] == "seed"]
    assert len(seed_rows) == 2
