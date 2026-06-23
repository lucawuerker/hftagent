"""Position-construction regime for deployed strategies.

Two regimes turn a strategy's composite signal into positions:
  * cross_sectional — rank the cross-section → dollar-neutral long/short (default);
  * per_underlying  — directional boundary per name, each active position sized
                      1/max_positions (default for LOBSTER).

These tests cover the shared primitives, the provider-default resolution + override,
the backtester dispatch, and that the regime survives persist → reload (so OOS /
live reproduce the same book).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting import positions as P
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.config import DataSettings, Settings
from quant_fund_agent.strategies.dynamic import DynamicStrategy


# ── shared primitives ────────────────────────────────────────────────────────

def test_directional_positions_threshold_band():
    z = pd.DataFrame({"A": [2.0, 0.0, -2.0], "B": [-1.5, 0.5, 1.5]})
    pos = P.directional_positions(z, mode="threshold", threshold=1.0)
    assert pos["A"].tolist() == [1.0, 0.0, -1.0]
    assert pos["B"].tolist() == [-1.0, 0.0, 1.0]


def test_per_underlying_positions_equal_weight_sizing():
    # n_max_positions=4 → each active name sized ±1/4; flat in the band.
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.standard_normal((n, 3)), index=idx, columns=["A", "B", "C"])
    pos = P.per_underlying_positions(
        sig, n_max_positions=4, mode="threshold", zscore_basis="full")
    vals = set(np.round(np.unique(pos.to_numpy()), 6).tolist())
    assert vals <= {-0.25, 0.0, 0.25}
    # a single long signal commits only 1/n_max_positions, never the whole book.
    assert float(pos.abs().to_numpy().max()) == 0.25


# ── provider default + override ──────────────────────────────────────────────

def test_default_construction_by_provider():
    lob = Settings(data=DataSettings(provider="lobster"))
    yf = Settings(data=DataSettings(provider="yfinance"))
    assert P.default_position_construction(lob) == "per_underlying"
    assert P.default_position_construction(yf) == "cross_sectional"


def test_resolve_precedence(monkeypatch):
    lob = Settings(data=DataSettings(provider="lobster"))
    # explicit override beats everything
    assert P.resolve_position_construction("cross_sectional", lob) == "cross_sectional"
    # "auto" → provider default
    assert P.resolve_position_construction("auto", lob) == "per_underlying"
    # env var when no explicit override
    monkeypatch.setenv("QF_POSITION_CONSTRUCTION", "per_underlying")
    yf = Settings(data=DataSettings(provider="yfinance"))
    assert P.resolve_position_construction(None, yf) == "per_underlying"


# ── backtester dispatch ──────────────────────────────────────────────────────

def _panel(tickers, n=200, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.standard_normal((n, len(tickers))) * 0.01, 0),
                         index=idx, columns=tickers)
    return {"close": close}


def test_backtest_dispatch_per_underlying_vs_cross_sectional():
    tickers = ["A", "B", "C", "D", "E"]
    panel = _panel(tickers)
    fsig = {"f1": panel["close"].pct_change(5).fillna(0.0)}

    cs = DynamicStrategy("s", "s", "", ["f1"], {"f1": 1.0}, max_positions=4)
    pu = DynamicStrategy("s", "s", "", ["f1"], {"f1": 1.0}, max_positions=4,
                         position_construction="per_underlying",
                         position_params={"zscore_basis": "full"})

    pos_cs = backtest_strategy(cs, fsig, panel).positions
    pos_pu = backtest_strategy(pu, fsig, panel).positions

    # cross-sectional rescales each bar to gross ≈ 1 (Σ|w| = 1)
    gross_cs = pos_cs.abs().sum(axis=1).replace(0, np.nan).dropna()
    assert np.allclose(gross_cs, 1.0)
    # per-underlying uses fixed ±1/max_positions weights (never rescaled to 1)
    vals = set(np.round(np.unique(pos_pu.to_numpy()[~np.isnan(pos_pu.to_numpy())]), 6))
    assert vals <= {-0.25, 0.0, 0.25}


# ── persistence round-trip (regime survives reload) ──────────────────────────

def test_strategy_from_record_restores_regime():
    from quant_fund_agent import pipeline
    from quant_fund_agent.schemas import StrategyRecord

    rec = StrategyRecord(
        id="s1", name="s1", model_type="static_weights",
        factor_ids=["f1"],
        model_params={
            "model_type": "static_weights",
            "weights": {"f1": 1.0},
            "holding_period": 6, "max_positions": 8,
            "position_construction": "per_underlying",
            "position_params": {"mode": "sign", "zscore_basis": "expanding"},
        },
    )
    strat = pipeline.strategy_from_record(rec)
    assert strat.position_construction == "per_underlying"
    assert strat.position_params == {"mode": "sign", "zscore_basis": "expanding"}
    assert strat.max_positions == 8


def test_strategy_from_record_defaults_cross_sectional():
    # A legacy record with no position_construction reloads as cross_sectional.
    from quant_fund_agent import pipeline
    from quant_fund_agent.schemas import StrategyRecord

    rec = StrategyRecord(
        id="s2", name="s2", model_type="static_weights", factor_ids=["f1"],
        model_params={"model_type": "static_weights", "weights": {"f1": 1.0}},
    )
    strat = pipeline.strategy_from_record(rec)
    assert strat.position_construction == "cross_sectional"
