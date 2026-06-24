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


# ── per-factor prediction horizons ─────────────────────────────────────────

def _ic_panel(n=500, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="10s")
    cols = [f"T{i}" for i in range(k)]
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0),
                         index=idx, columns=cols)
    sig = pd.DataFrame(rng.normal(0, 1, (n, k)), index=idx, columns=cols)
    return {"close": close}, sig


def test_backtest_factor_horizon_none_is_byte_identical():
    from quant_fund_agent.backtesting.engine import backtest_factor

    panel, sig = _ic_panel()
    legacy = backtest_factor(None, panel, horizons=(1, 6, 60), signal=sig)
    again = backtest_factor(None, panel, horizons=(1, 6, 60), signal=sig, factor_horizon=None)
    assert legacy.model_dump() == again.model_dump()


def test_backtest_factor_horizon_anchors_and_unions():
    from quant_fund_agent.backtesting.engine import backtest_factor

    panel, sig = _ic_panel()
    m = backtest_factor(None, panel, horizons=(1, 6, 60), signal=sig, factor_horizon=12)
    assert "12" in m.ic_by_horizon
    assert m.extra["horizon_bars"] == 12
    assert m.information_coefficient == m.ic_by_horizon["12"]["ic"]
    assert sorted(int(k) for k in m.ic_by_horizon) == [1, 6, 12, 60]


def test_resolve_target_horizon_aggregators():
    from quant_fund_agent.comparison.config import ComparisonConfig

    assert ComparisonConfig(preruns=["x"]).resolve_target_horizon([1, 1, 6, 60, 60, 60]) == 60
    assert ComparisonConfig(preruns=["x"], combined_horizon_agg="median") \
        .resolve_target_horizon([1, 6, 60]) == 6
    assert ComparisonConfig(preruns=["x"], combined_horizon_agg="max") \
        .resolve_target_horizon([1, 6, 60]) == 60
    cfg = ComparisonConfig(preruns=["x"], target_horizon=7, combined_horizon_agg="explicit")
    assert cfg.resolve_target_horizon([1, 1, 1]) == 7
    assert cfg.resolve_target_horizon([]) == 7
    ov = ComparisonConfig(preruns=["x"], target_horizon=9, target_horizon_overridden=True)
    assert ov.resolve_target_horizon([60, 60]) == 9
    assert ComparisonConfig(preruns=["x"]).resolve_target_horizon([60, 60], n_bars=5) == 4


def test_architect_chooses_then_inherits_horizon():
    from quant_fund_agent.agents.architect.graph import _default_target_horizon, _parse_spec
    from quant_fund_agent.agents.architect.state import ArchitectState
    from quant_fund_agent.agents.selector.state import FactorSummary

    cat = [FactorSummary(factor_id=i, name=i, category="c", description="d",
                         prediction_horizon=h)
           for i, h in [("a", 60), ("b", 60), ("c", 6)]]
    st = ArchitectState(hypothesis="h", selected_factor_ids=["a", "b", "c"],
                        factor_catalog=cat, target_horizon=6)
    assert _default_target_horizon(st) == 60

    spec = _parse_spec({"model_type": "linear_regression"}, st, "x")
    assert spec.target_horizon == 60 and spec.holding_period == 60
    spec2 = _parse_spec({"model_type": "ridge", "target_horizon": 20}, st, "x")
    assert spec2.target_horizon == 20

    st.strategy_spec = spec2.model_copy(update={"target_horizon": 33})
    spec3 = _parse_spec({"model_type": "ridge"}, st, "x", inherit_horizon=True)
    assert spec3.target_horizon == 33


def test_data_context_bar_size_and_horizon_contract():
    from quant_fund_agent.agents.factor_research.prompts import (
        _bar_size_phrase,
        build_data_context,
    )

    assert _bar_size_phrase(10) == "10-second"
    assert _bar_size_phrase(60) == "1-minute"
    assert _bar_size_phrase(86400) == "daily"
    assert _bar_size_phrase(None) is None

    fields = ["open", "high", "low", "close", "volume"]
    ctx10 = build_data_context(fields, seconds_per_bar=10)
    assert "PREDICTION HORIZON" in ctx10 and "10 second" in ctx10
    assert "daily" in build_data_context(fields, seconds_per_bar=86400)
    ctx_none = build_data_context(fields, seconds_per_bar=None)
    assert "PREDICTION HORIZON" in ctx_none
    assert "10-second" not in ctx_none
