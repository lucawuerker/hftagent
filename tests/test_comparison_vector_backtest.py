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


# ── fit_scope: pooled (one model) vs per_underlying (a model per name) ────────

def test_fit_scope_per_underlying_vs_pooled_diverge():
    """Two names with OPPOSITE factor→return signs: a pooled fit cancels to ~0,
    while a per-underlying fit recovers +1 for one name and -1 for the other."""
    cols = pd.Index(["A", "B"])
    n_cols, T = 2, 200
    index = pd.date_range("2024-01-01", periods=T, freq="min")
    rng = np.random.default_rng(0)

    # Timestamp-major rows: row r → (t = r // n_cols, j = r % n_cols).  One feature;
    # target is +x for A (j=0) and -x for B (j=1).
    x = rng.standard_normal(T * n_cols)
    X = x.reshape(-1, 1)
    j = np.arange(T * n_cols) % n_cols
    y = np.where(j == 0, x, -x)
    t = np.arange(T * n_cols) // n_cols
    train = np.flatnonzero(t < int(T * 0.7))            # IS rows

    pooled = bruteforce._combined_signal(
        "ridge", X, y, train, index, cols, n_cols, _cfg(fit_scope="pooled"))
    per = bruteforce._combined_signal(
        "ridge", X, y, train, index, cols, n_cols, _cfg(fit_scope="per_underlying"))

    assert pooled.shape == per.shape == (T, n_cols)
    xA, xB = x[0::2], x[1::2]
    # per-underlying recovers the opposite signs per name …
    assert np.corrcoef(per["A"].to_numpy(), xA)[0, 1] > 0.9
    assert np.corrcoef(per["B"].to_numpy(), xB)[0, 1] < -0.9
    # … while the single pooled model sees them cancel → a near-flat signal.
    assert per["A"].std() > 5 * (pooled["A"].std() + 1e-12)


def test_fit_scope_per_underlying_needs_enough_rows():
    """A per-underlying fit with too few IS rows for every name raises (the caller
    turns this into a degraded 'error' row rather than a crash)."""
    cols = pd.Index(["A", "B"])
    n_cols, T = 2, 50
    index = pd.date_range("2024-01-01", periods=T, freq="min")
    X = np.random.default_rng(1).standard_normal((T * n_cols, 1))
    y = X.ravel()
    train = np.arange(2 * n_cols)                       # 2 IS rows per name (< minimum)
    with pytest.raises(ValueError, match="per-underlying fit"):
        bruteforce._combined_signal(
            "ridge", X, y, train, index, cols, n_cols, _cfg(fit_scope="per_underlying"))


# ── calendar (date-window) IS/OOS split ──────────────────────────────────────

def _multi_month_panel(tickers, seed=0):
    """An hourly panel spanning Jun–Sep 2024 (so month-based splits are meaningful)."""
    idx = pd.date_range("2024-06-01", "2024-09-30 23:00", freq="h")
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((len(idx), len(tickers))) * 0.01
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=tickers)
    return {"close": close}, idx


def test_period_mask_formats():
    from quant_fund_agent.comparison.config import period_mask

    idx = pd.date_range("2024-06-01", "2024-09-30 23:00", freq="h")
    assert set(idx[period_mask("2024-06,2024-07", idx)].month) == {6, 7}
    # a month at the END of a range expands to the whole month (all of August)
    aug = idx[period_mask("2024-06:2024-08", idx)]
    assert set(aug.month) == {6, 7, 8} and aug.max().day == 31
    day = idx[period_mask("2024-09-01:2024-09-15", idx)]
    assert day.min() == pd.Timestamp("2024-09-01 00:00")
    assert day.max() == pd.Timestamp("2024-09-15 23:00")


def test_calendar_split_drives_is_and_oos():
    tickers = ["A", "B", "C", "D"]
    panel, idx = _multi_month_panel(tickers)
    cfg = _cfg(target_horizon=1, is_window="2024-06:2024-08", oos_window="2024-09",
               position_mode="sign", position_zscore_basis="none")

    is_mask, oos_mask = cfg.split_masks(idx)
    assert set(idx[is_mask].month) == {6, 7, 8}
    assert set(idx[oos_mask].month) == {9}
    assert int((is_mask & oos_mask).sum()) == 0

    # a clairvoyant signal still scores on the OOS (September) slice only
    out = vb.vector_backtest(panel["close"].pct_change().shift(-1), panel, cfg)
    assert out["oos_sharpe"] is not None and out["oos_sharpe"] > 1.0

    # the model's pooled training rows fall exclusively in the IS months
    service._SIGNAL_CACHE = {"f1": panel["close"].pct_change()}
    X, y, train, index, cols, n_cols = bruteforce._pooled_features(["f1"], panel, cfg)
    train_months = set(pd.DatetimeIndex(index[train // n_cols]).month)
    assert train_months <= {6, 7, 8}


def test_calendar_split_validation():
    idx = pd.date_range("2024-06-01", "2024-09-30 23:00", freq="h")
    with pytest.raises(ValueError, match="BOTH"):           # only one window given
        _cfg(is_window="2024-06").split_masks(idx)
    with pytest.raises(ValueError, match="overlap"):        # windows share August
        _cfg(is_window="2024-06:2024-08", oos_window="2024-08:2024-09").split_masks(idx)
