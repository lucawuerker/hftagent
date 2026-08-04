"""Strategy compiler — pipeline invariants on a synthetic panel.

Covers the risk-layer guarantees the product depends on: turnover falls with
smoothing/band, concentrated books respect max_positions with entry/exit
hysteresis, net exposure lands where the persona asked, vol targeting scales
toward the target, beta neutralisation kills the market bet, and personas load
from personas.yaml with the theme filter selecting by category.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.backtesting.strategy_compiler import (
    MODEL_BLENDS,
    Persona,
    RiskParams,
    blend_predictions,
    compile_positions,
    cs_zscore,
    load_personas,
    strategy_returns,
)

N_BARS, N_NAMES = 600, 40
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def close() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=N_BARS, freq="B")
    rets = RNG.normal(0.0004, 0.015, size=(N_BARS, N_NAMES))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                        columns=[f"T{i:02d}" for i in range(N_NAMES)])


@pytest.fixture(scope="module")
def pred(close) -> pd.DataFrame:
    # persistent-ish signal with noise (autocorrelated, like a 6-bar forecast)
    base = RNG.normal(size=(N_BARS, N_NAMES))
    sig = pd.DataFrame(base, index=close.index, columns=close.columns)
    return sig.ewm(halflife=10).mean() + 0.1 * RNG.normal(size=(N_BARS, N_NAMES))


def _risk(**kw) -> RiskParams:
    defaults = dict(beta_neutral=False, sector_neutral=False)
    defaults.update(kw)
    return RiskParams(**defaults)


def test_smoothing_and_band_reduce_turnover(pred, close):
    fast = compile_positions(pred, close, _risk(halflife=1, band=0.0))
    slow = compile_positions(pred, close, _risk(halflife=12, band=0.25))
    _, t_fast, _ = strategy_returns(fast, close)
    _, t_slow, _ = strategy_returns(slow, close)
    assert t_slow.iloc[50:].mean() < 0.5 * t_fast.iloc[50:].mean()


def test_max_positions_and_hysteresis(pred, close):
    w = compile_positions(pred, close, _risk(max_positions=10, exit_buffer=1.5))
    n_open = (w != 0).sum(axis=1)
    # never more than N·exit_buffer names held
    assert n_open.iloc[50:].max() <= 15
    assert n_open.iloc[50:].median() >= 5
    # hysteresis: fewer name entries/exits than a hard top-10 cutoff
    def churn(frame):
        held = frame != 0
        return (held != held.shift(1)).sum(axis=1).iloc[50:].mean()
    hard = compile_positions(pred, close, _risk(max_positions=10, exit_buffer=1.0))
    assert churn(w) <= churn(hard)


def test_net_exposure_lands_near_target(pred, close):
    w = compile_positions(pred, close, _risk(net_exposure=0.3, band=0.0))
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1)
    frac = (net / gross.replace(0, np.nan)).iloc[100:]
    assert 0.15 < frac.median() < 0.45
    w0 = compile_positions(pred, close, _risk(net_exposure=0.0, band=0.0))
    frac0 = (w0.sum(axis=1) / w0.abs().sum(axis=1).replace(0, np.nan)).iloc[100:]
    assert abs(frac0.median()) < 0.05


def test_vol_targeting_scales_toward_target(pred, close):
    lo = compile_positions(pred, close, _risk(vol_target=0.05))
    hi = compile_positions(pred, close, _risk(vol_target=0.20))
    n_lo, _, _ = strategy_returns(lo, close)
    n_hi, _, _ = strategy_returns(hi, close)
    v_lo = n_lo.iloc[100:].std() * np.sqrt(252)
    v_hi = n_hi.iloc[100:].std() * np.sqrt(252)
    assert v_hi > 1.5 * v_lo


def test_beta_neutral_kills_market_bet():
    # heterogeneous betas + a prediction that ranks names BY beta: the raw
    # LS book is then a high-beta-long/low-beta-short market bet, which the
    # beta projection must collapse
    idx = pd.date_range("2020-01-01", periods=600, freq="B")
    betas = np.linspace(0.5, 1.5, N_NAMES)
    f_m = RNG.normal(0.0, 0.012, size=600)
    eps = RNG.normal(0.0, 0.006, size=(600, N_NAMES))
    rets = betas[None, :] * f_m[:, None] + eps
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                      columns=[f"T{i:02d}" for i in range(N_NAMES)])
    pred_beta = pd.DataFrame(np.tile(betas, (600, 1)), index=idx,
                             columns=px.columns)
    pred_beta += 0.05 * RNG.normal(size=pred_beta.shape)
    mkt = px.pct_change().mean(axis=1)
    w_raw = compile_positions(pred_beta, px, _risk(band=0.0))
    w_neu = compile_positions(pred_beta, px, _risk(band=0.0, beta_neutral=True))
    r_raw, _, _ = strategy_returns(w_raw, px)
    r_neu, _, _ = strategy_returns(w_neu, px)
    c_raw = r_raw.iloc[150:].corr(mkt.shift(-1).iloc[150:-1])
    c_neu = r_neu.iloc[150:].corr(mkt.shift(-1).iloc[150:-1])
    assert abs(c_raw) > 0.5           # the raw book really is a market bet
    assert abs(c_neu) < 0.6 * abs(c_raw)


def test_blend_predictions_weighted_zscore():
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    cols = list("ABCD")
    a = pd.DataFrame(RNG.normal(size=(50, 4)), index=idx, columns=cols)
    b = pd.DataFrame(RNG.normal(size=(50, 4)) * 100, index=idx, columns=cols)
    out = blend_predictions({"random_forest": a, "lightgbm": b},
                            MODEL_BLENDS["rf_gbm"])
    # scale-free: the 100x model must not dominate beyond its blend weight
    za, zb = cs_zscore(a), cs_zscore(b)
    expect = 0.6 * za + 0.4 * zb
    pd.testing.assert_frame_equal(out, expect)


def test_personas_load_and_theme_filter():
    personas = load_personas()
    keys = {q.key for q in personas}
    assert {"master", "aggressive_short_term", "defensive_low_turnover",
            "earnings_events", "diversified_all_weather",
            "contrarian_dip_buyer"} <= keys
    master = next(q for q in personas if q.key == "master")
    assert master.risk.max_positions is None
    retail = [q for q in personas if q.key != "master"]
    assert all(q.risk.max_positions and q.risk.max_positions <= 20 for q in retail)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from simulate_user_strategies import select_factors
    rows = [{"factor_id": "f_mom", "category": "momentum", "trading_idea": "",
             "name": "", "mechanism": "", "score": 0.5},
            {"factor_id": "f_rev", "category": "mean_reversion",
             "trading_idea": "fade overreaction snapback", "name": "",
             "mechanism": "", "score": 0.4}]
    contrarian = next(q for q in personas if q.key == "contrarian_dip_buyer")
    picked = {r["factor_id"]
              for r in select_factors({"factors": rows}, contrarian.theme())}
    assert picked == {"f_rev"}


def test_deterministic(pred, close):
    r = _risk(max_positions=12)
    w1 = compile_positions(pred, close, r)
    w2 = compile_positions(pred, close, r)
    pd.testing.assert_frame_equal(w1, w2)


def test_net_exposure_comes_from_picking_not_index(pred, close):
    # ν=1: pure long-only picking book — no shorts at all
    w = compile_positions(pred, close, _risk(net_exposure=1.0, band=0.0))
    assert (w.iloc[100:] < -1e-12).sum().sum() == 0
    # the long leg must follow the ranking, not the universe: correlation of
    # weight with the processed score is positive, and names outside the top
    # half hold far less weight than the top half
    z = cs_zscore(pred).ewm(halflife=6, min_periods=1).mean()
    last = w.iloc[-2]
    top = z.iloc[-2].nlargest(10).index
    bottom = z.iloc[-2].nsmallest(10).index
    assert last[top].sum() > 3 * max(last[bottom].sum(), 0.0)
    # ν=0.6: net/gross lands near 0.6 (pre vol-targeting distortions)
    w6 = compile_positions(pred, close, _risk(net_exposure=0.6, band=0.0))
    frac = (w6.sum(axis=1) / w6.abs().sum(axis=1).replace(0, np.nan)).iloc[100:]
    assert 0.45 < frac.median() < 0.75
