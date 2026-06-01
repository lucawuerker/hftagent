"""Tests for horizon plumbing + validation-based trial selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.agents.architect.graph import _best_trial, _trial_score
from quant_fund_agent.agents.architect.state import (
    ArchitectState,
    StrategySpec,
    TrialRecord,
)
from quant_fund_agent.modeling import fit_and_backtest

FACTOR_IDS = ["f1", "f2"]


def _panel(n_bars=500, n_tickers=8):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="10s")
    cols = [f"T{i}" for i in range(n_tickers)]
    signals = {
        fid: pd.DataFrame(rng.normal(size=(n_bars, n_tickers)), index=idx, columns=cols)
        for fid in FACTOR_IDS
    }
    close = pd.DataFrame(
        100 + np.cumsum(rng.normal(0, 0.01, (n_bars, n_tickers)), axis=0),
        index=idx, columns=cols,
    )
    return signals, {"close": close}


def test_best_trial_selects_by_validation_ic():
    """The held-out validation IC wins selection even when IS Sharpe disagrees."""
    a = TrialRecord(iteration=1, spec=StrategySpec(strategy_name="A", model_type="ridge"),
                    metrics={"sharpe_ratio": 5.0, "validation_score": 0.01})
    b = TrialRecord(iteration=2, spec=StrategySpec(strategy_name="B", model_type="xgboost"),
                    metrics={"sharpe_ratio": 1.0, "validation_score": 0.05})
    best = _best_trial(ArchitectState(trial_history=[a, b]))
    assert best is not None and best.spec.strategy_name == "B"


def test_best_trial_falls_back_to_sharpe_for_static():
    """Static-weights trials have no validation score → fall back to IS Sharpe."""
    a = TrialRecord(iteration=1, spec=StrategySpec(model_type="static_weights"),
                    metrics={"sharpe_ratio": 2.0})
    b = TrialRecord(iteration=2, spec=StrategySpec(model_type="static_weights"),
                    metrics={"sharpe_ratio": 3.0})
    best = _best_trial(ArchitectState(trial_history=[a, b]))
    assert best is not None and best.metrics["sharpe_ratio"] == 3.0


def test_errored_trial_not_selected():
    a = TrialRecord(iteration=1, spec=StrategySpec(model_type="ridge"),
                    metrics={"error": "boom", "sharpe_ratio": None})
    assert _trial_score(a) is None


def test_fit_and_backtest_full_is_is_deterministic(tmp_path):
    """Refit-on-full-IS + ridge is deterministic → repeatable showcase numbers."""
    signals, data = _panel()
    kw = dict(factor_signals_is=signals, data_is=data, model_type="ridge",
              factor_ids=FACTOR_IDS, target_horizon=6, artifact_dir=tmp_path)
    r1 = fit_and_backtest(strategy_id="d1", **kw)
    r2 = fit_and_backtest(strategy_id="d2", **kw)
    assert r1["metrics"]["sharpe_ratio"] == r2["metrics"]["sharpe_ratio"]
    assert r1["validation_score"] == r2["validation_score"]


@pytest.mark.parametrize("horizon", [1, 6, 60])
def test_fit_and_backtest_runs_at_each_horizon(tmp_path, horizon):
    signals, data = _panel()
    r = fit_and_backtest(
        factor_signals_is=signals, data_is=data, model_type="ridge",
        factor_ids=FACTOR_IDS, target_horizon=horizon,
        holding_period=horizon, artifact_dir=tmp_path, strategy_id=f"h{horizon}",
    )
    assert r["metrics"]["sharpe_ratio"] is not None
    assert r["metrics"]["portfolio_returns"]["values"]
