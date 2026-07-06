"""Tests for the selection-time deflation *publish* filter (WS1).

The publish filter is the single choke point that enforces N_trials deflation on the
FINAL book (whatever produced it), replacing the old per-candidate search gate.  All
signal-based on a synthetic panel — no market data, no MCP.

Key properties under test:
* ``off`` mode keeps everything (discovery); ``on`` mode narrows (validation).
* Deflation acts on the book's COMBINED statistic, not any single factor's standalone
  IC, and pruning drops the LOWEST-MARGINAL member — so a complementary factor with
  weak standalone IC but strong marginal value is NOT dropped (correction #3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval.harness import _pooled_ic
from quant_fund_agent.research_eval.publish import publish_filter
from quant_fund_agent.research_eval.splits import three_way_split

N_BARS = 600
TICKERS = ["A", "B", "C", "D", "E"]


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}, rng, idx


def _frame(values, idx):
    return pd.DataFrame(values, index=idx, columns=TICKERS)


def _cfg(**kw):
    base = dict(preruns=["z"], target_horizon=1, fit_standardize="per_underlying", seed=0)
    base.update(kw)
    return ComparisonConfig(**base)


def _fwd(close):
    return close.pct_change().shift(-1)


# ── off mode / no-op ──────────────────────────────────────────────────────────

def test_off_mode_keeps_everything_even_under_many_trials():
    panel, rng, idx = _panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index)
    fwd = _fwd(panel["close"])
    sigs = {
        "good": fwd + 0.003 * rng.standard_normal((N_BARS, len(TICKERS))),
        "noise": _frame(rng.standard_normal((N_BARS, len(TICKERS))), idx),
    }
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=10_000, mode="off")
    assert set(res["kept_factor_ids"]) == {"good", "noise"}
    assert res["passed"] is True
    assert res["dropped"] == []


def test_single_trial_is_a_noop_for_a_real_edge():
    panel, rng, idx = _panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index)
    fwd = _fwd(panel["close"])
    sigs = {"good": fwd + 0.003 * rng.standard_normal((N_BARS, len(TICKERS)))}
    # n_trials=1 → no selection luck to subtract; a positive-IC book survives
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=1, mode="on")
    assert res["kept_factor_ids"] == ["good"]
    assert res["passed"] is True


# ── on mode: validation narrowing ─────────────────────────────────────────────

def test_on_mode_keeps_a_strong_book():
    panel, rng, idx = _panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index)
    fwd = _fwd(panel["close"])
    sigs = {"good": fwd + 0.003 * rng.standard_normal((N_BARS, len(TICKERS)))}
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=50, mode="on")
    assert res["kept_factor_ids"] == ["good"]
    assert res["passed"] is True


def test_on_mode_refuses_to_certify_pure_noise():
    panel, rng, idx = _panel(seed=1)
    cfg = _cfg()
    split = three_way_split(panel["close"].index)
    sigs = {f"n{i}": _frame(rng.standard_normal((N_BARS, len(TICKERS))), idx)
            for i in range(3)}
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=1_000_000, mode="on")
    # validation mode cannot certify a noise book — it never grows the book, and the
    # deflation verdict is a failure
    assert res["passed"] is False
    assert set(res["kept_factor_ids"]).issubset(set(sigs))
    assert len(res["kept_factor_ids"]) >= 1


def test_on_mode_drops_an_overfit_trap_but_keeps_the_real_factor():
    """A factor that predicts on IS but is noise on VAL has NEGATIVE marginal value
    (it lets the combiner overfit) → the publish filter prunes it, keeping the real
    (modest-standalone) factor.  Drops by marginal, not standalone IC."""
    panel, rng, idx = _panel(seed=2)
    cfg = _cfg()
    split = three_way_split(panel["close"].index)
    fwd = _fwd(panel["close"])

    good = fwd + 0.02 * rng.standard_normal((N_BARS, len(TICKERS)))  # modest real edge
    trap_vals = rng.standard_normal((N_BARS, len(TICKERS)))
    trap_vals[split.is_mask] = fwd.to_numpy()[split.is_mask]        # perfect on IS only
    trap = _frame(trap_vals, idx)

    sigs = {"good": good, "trap": trap}
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=500, mode="on")
    assert "good" in res["kept_factor_ids"]
    assert "trap" not in res["kept_factor_ids"]
    assert "trap" in res["dropped"]


# ── correction #3: a weak-standalone / strong-marginal factor is NOT dropped ───

def _interaction_panel(seed=7):
    """fwd = base · state (a conditioning/interaction panel).

    ``state`` flips the sign of ``base``'s predictive power bar by bar, so ``state``
    has ~0 STANDALONE IC yet huge MARGINAL value (a nonlinear combiner needs it to
    read ``base``).  The canonical vol-as-state-variable case.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    rets = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)

    base_v = rng.standard_normal((N_BARS, len(TICKERS)))
    state_v = np.where(rng.standard_normal((N_BARS, 1)) > 0, 1.0, -1.0) \
        * np.ones((1, len(TICKERS)))
    # define the forward return from base·state, then back it into a close path is
    # overkill — instead score against this synthetic target directly via a close whose
    # 1-bar forward return equals base·state (+noise).
    target = base_v * state_v + 0.1 * rng.standard_normal((N_BARS, len(TICKERS)))
    # build a close whose pct_change().shift(-1) ≈ target: close[t+1]/close[t]-1 = target[t]
    growth = 1.0 + 0.01 * target
    close = pd.DataFrame(100 * np.cumprod(np.vstack([np.ones((1, len(TICKERS))),
                                                     growth[:-1]]), axis=0),
                         index=idx, columns=TICKERS)
    base = pd.DataFrame(base_v, index=idx, columns=TICKERS)
    state = pd.DataFrame(state_v, index=idx, columns=TICKERS)
    return {"close": close}, base, state, idx


def test_weak_standalone_strong_marginal_factor_is_retained():
    panel, base, state, idx = _interaction_panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index)

    # `state` has ~0 standalone IC (its predictive power only exists via base·state)
    state_ic = _pooled_ic(state, panel["close"], cfg.target_horizon,
                          split.val_mask, split.is_mask, split.is_val_mask)[0]
    assert abs(state_ic or 0.0) < 0.1

    sigs = {"base": base, "state": state}
    # gradient boosting reads the interaction, so the book has a real combined edge;
    # even in strict validation mode the low-standalone `state` must NOT be dropped
    res = publish_filter(sigs, panel["close"], cfg, split, n_trials=20, mode="on",
                         marginal_model="gradient_boosting")
    assert "state" in res["kept_factor_ids"]
    assert "base" in res["kept_factor_ids"]
