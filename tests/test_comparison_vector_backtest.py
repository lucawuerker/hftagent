"""Tests for the per-underlying vectorised backtest + the reworked ML-combine track.

All on a tiny synthetic panel (no LOBSTER on disk).  The ML track reads factor
signals from the modeling-service cache, which we pre-populate via monkeypatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison import bruteforce, vector_backtest as vb
from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.modeling import service

N_BARS = 400


def _frame(values, tickers):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="min")
    return pd.DataFrame(values, index=idx, columns=tickers)


def _panel(tickers, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((N_BARS, len(tickers))) * 0.01
    close = _frame(100 * np.cumprod(1 + rets, axis=0), tickers)
    return {"close": close}, rng


def _cfg(**kw):
    base = dict(preruns=["z"], target_horizon=1, oos_split_ratio=0.3, seed=0)
    base.update(kw)
    return ComparisonConfig(**base)


# ── vector_backtest ──────────────────────────────────────────────────────────

def test_perfect_signal_beats_noise():
    tickers = ["A", "B", "C", "D", "E"]
    panel, rng = _panel(tickers)
    fwd = panel["close"].pct_change().shift(-1)  # next-bar return = target_horizon 1

    cfg = _cfg(position_mode="sign", position_zscore_basis="none")
    perfect = vb.vector_backtest(fwd, panel, cfg)            # signal == next-bar return
    noise = vb.vector_backtest(_frame(rng.standard_normal((N_BARS, len(tickers))), tickers),
                               panel, cfg)
    assert perfect["oos_sharpe"] is not None
    assert perfect["oos_sharpe"] > 1.0          # a clairvoyant signal is very profitable
    assert perfect["oos_sharpe"] > (noise["oos_sharpe"] or 0.0)


def test_position_modes_change_positions():
    tickers = ["A", "B", "C"]
    panel, rng = _panel(tickers)
    sig = _frame(rng.standard_normal((N_BARS, len(tickers))), tickers)

    z = vb._zscore(sig, "full", 500)
    thr = vb._positions(z, _cfg(position_mode="threshold", position_threshold=1.0))
    sgn = vb._positions(z, _cfg(position_mode="sign"))
    cont = vb._positions(z, _cfg(position_mode="continuous"))

    assert set(np.unique(thr.values)) <= {-1.0, 0.0, 1.0}
    assert (thr.values == 0.0).any()                        # threshold band has flats
    assert not ((sgn.values == 0.0).all())                  # sign rarely flat
    assert ((cont.values > 0) & (cont.values < 1)).any()    # continuous is real-valued


def test_expanding_zscore_has_no_lookahead():
    tickers = ["A", "B"]
    panel, rng = _panel(tickers)
    sig = _frame(rng.standard_normal((N_BARS, len(tickers))), tickers)
    z = vb._zscore(sig, "expanding", 500)
    assert z.iloc[0].isna().all()   # first row can't be standardised from the past


def test_aggregation_modes():
    tickers = ["A", "B", "C", "D", "E"]
    panel, rng = _panel(tickers)
    sig = _frame(rng.standard_normal((N_BARS, len(tickers))), tickers)
    port = vb.vector_backtest(sig, panel, _cfg(backtest_aggregation="portfolio"))
    per = vb.vector_backtest(sig, panel, _cfg(backtest_aggregation="per_underlying"))
    assert port["oos_sharpe"] is not None
    assert per["oos_sharpe"] is not None
    assert per["oos_sharpe_std"] is not None   # spread across underlyings reported


def test_constant_signal_reports_none_not_zero():
    """A degenerate (constant) signal takes no positions → metrics are None, not 0.0."""
    tickers = ["A", "B", "C"]
    panel, _ = _panel(tickers)
    const = _frame(np.ones((N_BARS, len(tickers))), tickers)  # zero variance everywhere
    out = vb.vector_backtest(const, panel, _cfg(position_mode="threshold"))
    assert out["oos_sharpe"] is None
    assert out["is_sharpe"] is None


def test_works_with_single_ticker():
    """The original failure: 1 underlying. A per-underlying backtest must still work."""
    tickers = ["ONLY"]
    panel, _ = _panel(tickers)
    fwd = panel["close"].pct_change().shift(-1)
    out = vb.vector_backtest(fwd, panel, _cfg(position_mode="sign", position_zscore_basis="none"))
    assert out["oos_sharpe"] is not None
    assert out["oos_sharpe"] > 0.5             # clairvoyant signal on one asset still profits


# ── ML-combine track end-to-end ──────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_ml_track_recovers_predictive_factor(monkeypatch):
    tickers = ["A", "B", "C", "D", "E"]
    panel, rng = _panel(tickers)
    fwd = panel["close"].pct_change().shift(-1).fillna(0.0)
    # one factor IS the next-bar return (predictive), two are noise.
    signals = {
        "good": fwd + 0.001 * rng.standard_normal((N_BARS, len(tickers))),
        "noise1": _frame(rng.standard_normal((N_BARS, len(tickers))), tickers),
        "noise2": _frame(rng.standard_normal((N_BARS, len(tickers))), tickers),
    }
    monkeypatch.setattr(service, "_SIGNAL_CACHE", dict(signals))

    cfg = _cfg(models=["ridge"], position_mode="sign", position_zscore_basis="none")
    rows = bruteforce.evaluate_prerun_models("z", list(signals), panel, cfg)
    ridge = next(r for r in rows if r["model"] == "ridge")
    assert "error" not in ridge
    assert ridge["oos_sharpe"] is not None
    assert ridge["oos_sharpe"] > 0.5           # the model should find the good factor
