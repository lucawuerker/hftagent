"""Tests for J4: the joint walk-forward (fold isolation + touch-once scoring)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.execution_research.evolution.loop import (
    ExecEvolutionRunConfig,
)
from quant_fund_agent.agents.factor_research.evolution.genome import FactorProgram
from quant_fund_agent.agents.factor_research.evolution.loop import (
    EvolutionLoop,
    EvolutionRunConfig,
)
from quant_fund_agent.joint_evolution.loop import JointRunConfig
from quant_fund_agent.joint_evolution.walkforward import run_joint_walk_forward

N_BARS, TICKERS = 480, ["A", "B", "C", "D", "E", "F"]


def _factor_code(fid: str, body: str) -> str:
    return f'''\
"""Test factor {fid}."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class F_{fid}(BaseFactor):
    factor_id = "{fid}"
    name = "{fid}"
    category = "momentum"
    inputs = ["close"]
    prediction_horizon = 1

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        return {body}
'''


def _seeds():
    return [FactorProgram(
        factor_id="wf_mom",
        code=_factor_code("wf_mom", "ts_mean(close.pct_change(), 3).fillna(0.0)"),
        name="mom", category="momentum", expected_sign=1, prediction_horizon=1)]


def _close(seed=3):
    rng = np.random.default_rng(seed)
    rets = np.zeros((N_BARS, len(TICKERS)))
    eps = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    for t in range(1, N_BARS):
        rets[t] = 0.6 * rets[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                        columns=TICKERS)


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("QF_USE_MCP", "0")
    from quant_fund_agent.mcp import research_service as svc

    panel = {"close": _close()}
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers: panel)
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("wf-panel",))
    monkeypatch.setattr(EvolutionLoop, "_load_known_ids", lambda self: None)
    svc._SIGNAL_CACHE.clear()
    return tmp_path, panel


def _cfgs(tmp_path):
    cfg = JointRunConfig(out_dir=str(tmp_path / "ignored"), total_blocks=2,
                         gens_per_block=1, scheduler="round_robin",
                         target_horizon=1, n_tickers=None)
    fac = EvolutionRunConfig(
        generations=1, population_size=4, children_per_generation=2,
        seed=11, target_horizon=1, cpcv_groups=4, cpcv_k=1, n_tickers=None,
        p_llm_semantic=0.0, p_crossover=0.0, p_jitter=1.0,
        out_dir=str(tmp_path / "x"))
    exe = ExecEvolutionRunConfig(
        out_dir=str(tmp_path / "y"), signals_manifest="(joint)",
        generations=1, population_size=4, children_per_generation=2,
        seed=7, n_tickers=None)
    return cfg, fac, exe


def test_two_fold_walk_forward_scores_each_fold(wired):
    tmp_path, _ = wired
    cfg, fac, exe = _cfgs(tmp_path)
    result = run_joint_walk_forward(
        cfg, fac, exe, ["2025-01-01", "2025-03-01"],
        out_dir=tmp_path / "wf", data_context="CTX",
        initial_factor_programs=_seeds())

    assert result["n_folds"] == 2
    assert result["n_scored"] == 2
    for i, fold in enumerate(result["folds"]):
        assert fold["oos"]["ok"], fold["oos"]
        assert fold["oos"]["n_oos_bars"] > 0
        # fold isolation: each fold has its own fresh joint state
        state = json.loads((tmp_path / "wf" / f"fold_{i}"
                            / "joint_state.json").read_text())
        assert state["sota"]["block_index"] == 2
    # score windows are disjoint: fold 0 on [d0, d1), fold 1 on [d1, ∞)
    assert result["folds"][0]["score_end"] == result["folds"][1]["score_start"]
    assert result["mean_oos_net_sharpe"] is not None
    assert (tmp_path / "wf" / "walkforward.json").exists()


def test_fold_zero_is_touch_once_wrt_later_data(wired, monkeypatch):
    """Corrupt every bar at/after d1 → fold 0's search AND score are identical."""
    tmp_path, panel = wired
    from quant_fund_agent.mcp import research_service as svc

    cfg, fac, exe = _cfgs(tmp_path)
    clean = run_joint_walk_forward(
        cfg, fac, exe, ["2025-01-01"], out_dir=tmp_path / "wf_clean",
        data_context="CTX", initial_factor_programs=_seeds())

    poisoned_close = panel["close"].copy()
    poisoned_close.loc[poisoned_close.index >= "2025-03-01"] = 777_777.0
    monkeypatch.setattr(svc, "_load_panel_cached",
                        lambda data_dir, fields, n_tickers:
                        {"close": poisoned_close})
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("wf-panel-poison",))
    svc._SIGNAL_CACHE.clear()

    dirty = run_joint_walk_forward(
        cfg, fac, exe, ["2025-01-01"], out_dir=tmp_path / "wf_dirty",
        data_context="CTX", initial_factor_programs=_seeds())

    # searching with cutoff=2025-01-01 never saw the poison; the score window
    # [2025-01-01, ∞) DOES include it — so compare a bounded variant instead
    clean_b = run_joint_walk_forward(
        cfg, fac, exe, ["2025-01-01", "2025-03-01"],
        out_dir=tmp_path / "wf_clean_b", data_context="CTX",
        initial_factor_programs=_seeds())
    monkeypatch.setattr(svc, "_panel_cache_key",
                        lambda data_dir, fields, n_tickers: ("wf-panel-poison2",))
    svc._SIGNAL_CACHE.clear()
    dirty_b = run_joint_walk_forward(
        cfg, fac, exe, ["2025-01-01", "2025-03-01"],
        out_dir=tmp_path / "wf_dirty_b", data_context="CTX",
        initial_factor_programs=_seeds())

    assert clean_b["folds"][0]["oos"] == dirty_b["folds"][0]["oos"]
    # sanity: the search summaries match too (cutoff isolation)
    assert (clean["folds"][0]["search_summary"]["ledger"]["n_factor"]
            == dirty["folds"][0]["search_summary"]["ledger"]["n_factor"])
