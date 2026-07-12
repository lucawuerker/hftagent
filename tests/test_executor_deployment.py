"""E4: executor_id deployment seam — registry resolution + backtester dispatch.

The load-bearing guarantees: (1) executor_id=None is byte-identical to the
legacy pipelines; (2) a registered executor reproduces the identical book in
`backtest_strategy` (the seam Architect IS-fit / Statistician OOS / live share);
(3) resolution: override > QF_EXECUTOR > None, unknown ids fail fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.execution.base import resolve_executor
from quant_fund_agent.strategies.dynamic import DynamicStrategy

T, N = 300, 6
TICKERS = [f"S{i}" for i in range(N)]


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="D")
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0),
                         index=idx, columns=TICKERS)
    return {"close": close}


def _signals(seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=T, freq="D")
    return {"f1": pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=TICKERS)}


def _strategy(**kw):
    base = dict(strategy_id="s1", name="s1", description="",
                factor_ids=["f1"], weights={"f1": 1.0}, holding_period=1)
    base.update(kw)
    return DynamicStrategy(**base)


# ── resolution ─────────────────────────────────────────────────────────────────

def test_resolve_executor_default_is_none(monkeypatch):
    monkeypatch.delenv("QF_EXECUTOR", raising=False)
    assert resolve_executor() is None
    assert resolve_executor("auto") is None
    assert resolve_executor("legacy") is None


def test_resolve_executor_env_and_override(monkeypatch):
    monkeypatch.setenv("QF_EXECUTOR", "zscore_threshold_equal_weight")
    assert resolve_executor() == "zscore_threshold_equal_weight"
    assert resolve_executor("topk_dollar_neutral") == "topk_dollar_neutral"
    with pytest.raises(KeyError, match="unknown executor_id"):
        resolve_executor("does_not_exist")


# ── backtester dispatch ────────────────────────────────────────────────────────

def test_executor_none_is_byte_identical_to_legacy():
    data, signals = _panel(), _signals()
    legacy = backtest_strategy(_strategy(), signals, data)
    explicit_none = backtest_strategy(_strategy(executor_id=None), signals, data)
    pd.testing.assert_frame_equal(legacy.positions, explicit_none.positions)
    pd.testing.assert_series_equal(legacy.portfolio_returns,
                                   explicit_none.portfolio_returns)


def test_executor_builds_its_own_book():
    """The executor path uses the executor's OWN params (self-contained),
    reproducing per_underlying_positions at the seed's defaults."""
    from quant_fund_agent.backtesting.positions import per_underlying_positions
    from quant_fund_agent.backtesting.strategy_backtester import (
        normalise_factor_signals,
    )

    data, signals = _panel(), _signals()
    res = backtest_strategy(
        _strategy(executor_id="zscore_threshold_equal_weight",
                  position_construction="per_underlying"),
        signals, data)

    signal = normalise_factor_signals(signals)["f1"] * 1.0  # weights {"f1": 1.0}
    want = per_underlying_positions(signal, n_max_positions=6)  # seed defaults
    pd.testing.assert_frame_equal(res.positions, want)


def test_executor_overrides_via_position_params():
    from quant_fund_agent.backtesting.positions import per_underlying_positions
    from quant_fund_agent.backtesting.strategy_backtester import (
        normalise_factor_signals,
    )

    data, signals = _panel(), _signals()
    res = backtest_strategy(
        _strategy(executor_id="zscore_threshold_equal_weight",
                  position_params={"executor_overrides": {"n_max_positions": 3,
                                                          "threshold": 0.5}}),
        signals, data)
    signal = normalise_factor_signals(signals)["f1"] * 1.0
    want = per_underlying_positions(signal, n_max_positions=3, threshold=0.5)
    pd.testing.assert_frame_equal(res.positions, want)


def test_spec_and_record_carry_executor_id():
    from quant_fund_agent.agents.architect.state import StrategySpec
    from quant_fund_agent.schemas import StrategyRecord

    spec = StrategySpec(executor_id="topk_dollar_neutral")
    assert spec.executor_id == "topk_dollar_neutral"
    assert StrategySpec().executor_id is None            # default = legacy

    rec = StrategyRecord(id="r1", name="r1", executor_id="topk_dollar_neutral")
    assert rec.executor_id == "topk_dollar_neutral"
    assert StrategyRecord(id="r2", name="r2").executor_id is None
