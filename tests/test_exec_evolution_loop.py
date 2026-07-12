"""Tests for the execution-arm evolution loop (E1): seeds, jitter, resume, rescore.

Everything runs in-process (QF_USE_MCP=0) on a synthetic AR(1) panel with the
service's panel loader monkeypatched — no market data, no LLM, no server.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.execution_research.evolution.genome import (
    ExecutionProgram,
)
from quant_fund_agent.agents.execution_research.evolution.loop import (
    ExecEvolutionLoop,
    ExecEvolutionRunConfig,
)
from quant_fund_agent.agents.execution_research.evolution.mutation import (
    jitter_params,
    jitter_variants,
    param_constants,
    random_jitter_child,
)
from quant_fund_agent.agents.execution_research.evolution.seeds import (
    TOPK_SEED_CODE,
    ZTHRESH_SEED_CODE,
    seed_execution_programs,
)
from quant_fund_agent.execution.codegen import compile_executor_inmem

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]

FACTOR_CODE = '''\
"""Test factor exz_mom."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FExzMom(BaseFactor):
    factor_id = "exz_mom"
    name = "exz_mom"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return close.pct_change().fillna(0.0)
'''


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


@pytest.fixture()
def frozen_manifest(synthetic_panel, tmp_path, monkeypatch):
    """Freeze one ridge evaluation signal from a 1-factor book, in-process."""
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("exec-test-panel",))
    svc._SIGNAL_CACHE.clear()
    out = svc.freeze_signals(
        [{"factor_id": "exz_mom", "code": FACTOR_CODE}],
        out_dir=str(tmp_path), version=1, target_horizon=1,
        specs=[{"model": "ridge", "subset": [0]}])
    assert out["ok"], out.get("error")
    return out["manifest_path"]


def _cfg(tmp_path, manifest, **kw) -> ExecEvolutionRunConfig:
    base = dict(
        out_dir=str(tmp_path / "evolution_exec"), signals_manifest=manifest,
        generations=2, population_size=6, children_per_generation=3,
        seed=7, n_tickers=None,
    )
    base.update(kw)
    return ExecEvolutionRunConfig(**base)


# ── seed byte-equivalence (genome #0 == the deployed baselines) ────────────────

def test_topk_seed_code_matches_legacy_pipeline():
    from quant_fund_agent.backtesting.strategy_backtester import _signal_to_positions

    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    signal = pd.DataFrame(rng.normal(0, 1, (200, 8)), index=idx,
                          columns=[f"S{i}" for i in range(8)])
    cls = compile_executor_inmem(TOPK_SEED_CODE, "seed_topk_dollar_neutral")
    pd.testing.assert_frame_equal(cls().target_weights(signal, {}),
                                  _signal_to_positions(signal))


def test_zthresh_seed_code_matches_legacy_pipeline():
    from quant_fund_agent.backtesting.positions import per_underlying_positions

    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    signal = pd.DataFrame(rng.normal(0, 1, (200, 8)), index=idx,
                          columns=[f"S{i}" for i in range(8)])
    cls = compile_executor_inmem(ZTHRESH_SEED_CODE, "seed_zscore_threshold")
    pd.testing.assert_frame_equal(
        cls().target_weights(signal, {}),
        per_underlying_positions(signal, n_max_positions=6))


# ── the param-jitter operator ──────────────────────────────────────────────────

def test_param_constants_extracts_the_jitter_surface():
    assert param_constants(TOPK_SEED_CODE) == {
        "holding_period": 1.0, "max_positions": 20.0, "min_conviction": 0.0}
    assert param_constants("x = 1") == {}


def test_jitter_params_scales_and_types():
    code, n = jitter_params(ZTHRESH_SEED_CODE, 1.5)
    assert n == 3                        # n_max 6→9, hp 1→2, threshold 1.0→1.5
    p = param_constants(code)
    assert p["n_max_positions"] == 9.0   # int stays int, rounded
    assert p["threshold"] == pytest.approx(1.5)
    assert p["holding_period"] == 2.0    # max(1, round(1 × 1.5)) = 2, still int

    down, n_down = jitter_params(ZTHRESH_SEED_CODE, 0.5)
    assert param_constants(down)["holding_period"] == 1.0  # floored at 1, never 0
    code_same, n_same = jitter_params(ZTHRESH_SEED_CODE, 1.0)
    assert n_same == 0                   # nothing moved → no fake variant


def test_jitter_child_compiles_and_differs():
    seed = seed_execution_programs()[1]
    child = random_jitter_child(seed, np.random.default_rng(0), "ex1_test_child",
                                pct=0.3)
    assert child is not None
    assert child.executor_id == "ex1_test_child"
    assert child.code != seed.code
    cls = compile_executor_inmem(child.code, "ex1_test_child")
    assert cls.executor_id == "ex1_test_child"


def test_jitter_variants_make_probe_ids():
    seed = seed_execution_programs()[0]
    variants = jitter_variants(seed, (0.5, 2.0))
    assert [vid for vid, _ in variants] == [
        "seed_topk_dollar_neutral_jit0", "seed_topk_dollar_neutral_jit1"]
    prog_no_params = ExecutionProgram(executor_id="none_test", code="x = 1")
    assert jitter_variants(prog_no_params) == []


# ── the loop end-to-end (in-process) ──────────────────────────────────────────

def test_loop_runs_checkpoints_and_finds_sota(tmp_path, frozen_manifest):
    loop = ExecEvolutionLoop(_cfg(tmp_path, frozen_manifest))
    summary = loop.run()

    assert summary["generations"] == 2
    assert summary["n_trials"] >= 2                  # both seeds scored
    assert summary["archive"], "seeds have real edge → archive must not be empty"
    assert summary["sota_executor"] is not None

    state_path = tmp_path / "evolution_exec" / "state.json"
    assert state_path.exists()
    lineage = [json.loads(l) for l in
               (tmp_path / "evolution_exec" / "lineage.jsonl").read_text().splitlines()]
    assert len(lineage) == summary["n_trials"]
    ops = {row["operator"] for row in lineage}
    assert "seed" in ops

    sota = loop.sota_executor()
    assert set(sota) >= {"executor_id", "code", "regime", "genome_id", "objective"}


def test_checkpoint_roundtrips_execution_programs(tmp_path, frozen_manifest):
    """Genome.program_type='executor' must reload as ExecutionProgram, not FactorProgram."""
    from quant_fund_agent.agents.factor_research.evolution.controller import (
        EvolutionController,
    )

    loop = ExecEvolutionLoop(_cfg(tmp_path, frozen_manifest, generations=1))
    loop.run()
    reloaded = EvolutionController.load(tmp_path / "evolution_exec" / "state.json")
    progs = [p for eg in reloaded.archive for p in eg.genome.programs]
    assert progs
    assert all(isinstance(p, ExecutionProgram) for p in progs)
    assert all(p.executor_id for p in progs)


def test_resume_continues_generations_and_trials(tmp_path, frozen_manifest):
    cfg = _cfg(tmp_path, frozen_manifest, generations=1)
    first = ExecEvolutionLoop(cfg)
    s1 = first.run()
    assert s1["generations"] == 1

    second = ExecEvolutionLoop(cfg)
    s2 = second.run(resume=True, n_generations=2)
    assert s2["generations"] == 3                     # 1 + 2 more
    assert s2["n_trials"] >= s1["n_trials"]           # counter carried over

    lineage = [json.loads(l) for l in
               (tmp_path / "evolution_exec" / "lineage.jsonl").read_text().splitlines()]
    seed_rows = [r for r in lineage if r["operator"] == "seed"]
    assert seed_rows and all(r["generation"] == 0 for r in seed_rows)
    assert len(lineage) == s2["n_trials"]             # old rows kept, not wiped


def test_rescore_archive_rebills_nothing_and_rebuilds(tmp_path, frozen_manifest,
                                                      synthetic_panel, monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: synthetic_panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("exec-test-panel",))

    loop = ExecEvolutionLoop(_cfg(tmp_path, frozen_manifest, generations=1))
    loop.run()
    n_trials_before = loop.controller.n_trials
    assert loop.controller.archive

    # a v2 freeze (same book, same spec → same signals, different version)
    v2 = svc.freeze_signals(
        [{"factor_id": "exz_mom", "code": FACTOR_CODE}],
        out_dir=str(tmp_path), version=2, target_horizon=1,
        specs=[{"model": "ridge", "subset": [0]}])
    assert v2["ok"]

    out = loop.rescore_archive(v2["manifest_path"])
    assert out["rescored"] > 0
    assert out["failed"] == 0
    assert out["archive"] > 0
    assert loop.controller.n_trials == n_trials_before   # looks, not hypotheses
    assert loop.cfg.signals_manifest == v2["manifest_path"]
    # the rescore event is auditable in the lineage
    events = [r for r in loop.controller.lineage if r.get("event") == "rescore_archive"]
    assert events and events[-1]["rescored"] == out["rescored"]
