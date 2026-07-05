"""Tests for the deterministic fitness harness (P0 foundation).

All on a small synthetic panel — no LOBSTER / market data required.  The harness is
signal-based, so we feed it factor signal frames directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval.harness import (
    EvalParams,
    evaluate_candidate,
    evaluate_set,
)
from quant_fund_agent.research_eval.splits import three_way_split

N_BARS = 600
TICKERS = ["A", "B", "C", "D", "E"]


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    return {"close": close}, rng


def _frame(values):
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    return pd.DataFrame(values, index=idx, columns=TICKERS)


def _cfg(**kw):
    base = dict(preruns=["z"], target_horizon=1, fit_standardize="per_underlying", seed=0)
    base.update(kw)
    return ComparisonConfig(**base)


def _noise(rng):
    return _frame(rng.standard_normal((N_BARS, len(TICKERS))))


def _poison_test_rows(frame, split, scale=1e9):
    out = frame.copy()
    rows = np.flatnonzero(split.test_mask)
    poison = scale * np.outer(
        np.arange(1, len(rows) + 1, dtype=float),
        np.arange(1, len(out.columns) + 1, dtype=float),
    )
    out.iloc[rows] = poison
    return out


def _assert_nested_close(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            _assert_nested_close(l_item, r_item)
    elif isinstance(left, (bool, str)) or left is None:
        assert left == right
    elif isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        assert int(left) == int(right)
    elif isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        assert float(left) == pytest.approx(float(right), rel=1e-12, abs=1e-12)
    else:
        assert left == right


# ── predictive candidate scores well; noise candidate does not ────────────────

def test_predictive_candidate_beats_noise():
    panel, rng = _panel()
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.002 * rng.standard_normal((N_BARS, len(TICKERS)))   # ~ next-bar return
    noise = _noise(rng)
    book = [_noise(rng)]

    params = EvalParams(n_trials=1, cpcv_groups=6, cpcv_k=2)
    good_r = evaluate_candidate(good, book, panel, cfg, params=params, candidate_id="good")
    noise_r = evaluate_candidate(noise, book, panel, cfg, params=params, candidate_id="noise")

    # standalone IC: strongly positive for the predictive one, ~0 for noise
    assert good_r.diagnostics["standalone_ic"] > 0.3
    assert abs(noise_r.diagnostics["standalone_ic"]) < 0.1
    # marginal value (LOCO ΔOOS-IC): the predictive factor adds real edge
    assert good_r.objective.marginal_value > noise_r.objective.marginal_value
    assert good_r.objective.marginal_value > 0


def test_predictive_candidate_passes_gates_noise_fails_under_many_trials():
    panel, rng = _panel()
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.002 * rng.standard_normal((N_BARS, len(TICKERS)))
    noise = _noise(rng)

    # under a large trial count the deflation gate should reject a lucky-noise factor
    params = EvalParams(n_trials=5000)
    good_r = evaluate_candidate(good, [], panel, cfg, params=params)
    noise_r = evaluate_candidate(noise, [], panel, cfg, params=params)

    assert good_r.gates.deflation_ok is True
    assert good_r.selectable
    assert not noise_r.selectable        # fails deflation and/or degradation


# ── independence axis (residual predictive content) ───────────────────────────

def _two_driver_panel(seed=0):
    """A panel whose forward return is driven by TWO orthogonal factors f1, f2.

    Lets the residual-IC independence axis discriminate: a factor spanning a
    *new* driver (f2) adds predictive content a book of f1 lacks, whereas a copy
    of f1 does not.  (On a single-driver panel every predictive signal is
    collinear, so residual IC can't tell them apart.)
    """
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal((N_BARS, len(TICKERS)))
    f2 = rng.standard_normal((N_BARS, len(TICKERS)))
    ret = np.zeros((N_BARS, len(TICKERS)))
    ret[1:] = 0.01 * (f1 + f2)[:-1]          # fwd[t] = ret[t+1] = 0.01·(f1[t]+f2[t])
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=TICKERS)
    return {"close": close}, _frame(f1), _frame(f2), rng


def test_independence_rewards_novel_predictive_content():
    panel, f1, f2, rng = _two_driver_panel()
    cfg = _cfg()
    book = [f1]                                              # book spans driver f1
    duplicate = f1 + 1e-6 * rng.standard_normal((N_BARS, len(TICKERS)))  # ~ copy of f1
    fresh = f2                                               # a NEW, orthogonal driver

    dup_r = evaluate_candidate(duplicate, book, panel, cfg, candidate_id="dup")
    fresh_r = evaluate_candidate(fresh, book, panel, cfg, candidate_id="fresh")

    # the diagnostics still see the redundancy (near-duplicate ~ collinear with book)
    assert dup_r.diagnostics["max_abs_corr"] > 0.9
    assert fresh_r.diagnostics["max_abs_corr"] < 0.3
    # residual-IC independence: the fresh driver has real edge orthogonal to the
    # book; the duplicate's edge is already spanned, so its residual IC is ~0.
    assert fresh_r.objective.independence == fresh_r.diagnostics["residual_ic"]
    assert fresh_r.objective.independence > 0.05
    assert fresh_r.objective.independence > dup_r.objective.independence


def test_independence_metric_delta_participation_is_selectable():
    panel, rng = _panel()
    cfg = _cfg()
    base = _noise(rng)
    duplicate = base + 1e-6 * rng.standard_normal((N_BARS, len(TICKERS)))
    fresh = _noise(rng)
    params = EvalParams(independence_metric="delta_participation")

    dup_r = evaluate_candidate(duplicate, [base], panel, cfg, params=params, candidate_id="dup")
    fresh_r = evaluate_candidate(fresh, [base], panel, cfg, params=params, candidate_id="fresh")
    # legacy Δ-participation axis: the unrelated factor raises the book's effective
    # dimensionality more than the near-duplicate does
    assert fresh_r.objective.independence > dup_r.objective.independence


# ── regime axis (crash-complementarity) ───────────────────────────────────────

def test_conditioning_factor_scores_positive_marginal_under_nonlinear_default():
    """A volatility-style *conditioning* factor (valuable only via an interaction,
    ~0 direct IC) must score a POSITIVE marginal value under the default nonlinear
    combiner — but ~0 under a linear ridge (which can't see the interaction)."""
    rng = np.random.default_rng(0)
    n_cols = len(TICKERS)
    idx = pd.date_range("2022-01-01", periods=N_BARS, freq="D")
    mom = rng.standard_normal((N_BARS, n_cols))            # a momentum-like driver
    vol01 = rng.uniform(0.0, 1.0, (N_BARS, n_cols))        # a volatility *state* in [0,1]
    # forward return = momentum whose payoff DECAYS with vol → pure interaction;
    # vol has ~0 main effect (E[fwd | vol] = E[mom]·(1-vol) = 0).
    fwd = mom * (1.0 - vol01)
    ret = np.zeros((N_BARS, n_cols)); ret[1:] = 0.01 * fwd[:-1]
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=TICKERS)
    panel = {"close": close}
    cfg = _cfg()
    split = three_way_split(idx, is_frac=0.6, val_frac=0.2)
    mom_f = pd.DataFrame(mom, index=idx, columns=TICKERS)
    vol_f = pd.DataFrame(vol01, index=idx, columns=TICKERS)

    # default EvalParams → gradient_boosting: the conditioning factor adds real edge
    nonlinear = evaluate_candidate(vol_f, [mom_f], panel, cfg, split=split,
                                   params=EvalParams(), candidate_id="vol")
    # ridge ablation: the same factor adds ~nothing (additive-only combiner)
    linear = evaluate_candidate(vol_f, [mom_f], panel, cfg, split=split,
                                params=EvalParams(marginal_model="ridge"),
                                candidate_id="vol")

    assert EvalParams().marginal_model == "gradient_boosting"        # nonlinear is the default
    assert nonlinear.objective.marginal_value > 0.01                 # conditioning value captured
    assert abs(linear.objective.marginal_value) < 0.01              # linear misses it
    assert nonlinear.objective.marginal_value > linear.objective.marginal_value


def test_regime_axis_rewards_crash_specialist():
    """A factor whose edge lives in the crash bars scores higher on the regime
    axis than one whose (equal-strength) edge lives only in the calm bars."""
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.research_eval.harness import _stress_mask

    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="min")
    # a common market factor (defines which bars are "crashes") + idiosyncratic noise
    mkt = rng.standard_normal(N_BARS) * 0.01
    ret = mkt[:, None] + rng.standard_normal((N_BARS, len(TICKERS))) * 0.01
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=TICKERS)
    panel = {"close": close}
    cfg = _cfg()
    split = three_way_split(idx, is_frac=0.5, val_frac=0.25)
    p = EvalParams(regime_kind="drawdown", regime_quantile=0.2, regime_min_obs=5)

    # build the two specialists directly off the harness's OWN crash labelling, so
    # "active in crashes" means exactly the bars the regime axis scores on.
    stress = _stress_mask(close, p, split.is_val_mask)              # (T,) bool
    stress_b = np.broadcast_to(stress[:, None], (N_BARS, len(TICKERS)))
    fwd = forward_returns(close, cfg.target_horizon).to_numpy(dtype=float)
    noise = rng.standard_normal((N_BARS, len(TICKERS)))
    crash_vals = np.where(stress_b, fwd, 1e-3 * noise)   # tracks fwd only in crashes
    calm_vals = np.where(stress_b, 1e-3 * noise, fwd)    # tracks fwd only in calm bars

    crash_r = evaluate_candidate(_frame(crash_vals), [], panel, cfg, split=split, params=p)
    calm_r = evaluate_candidate(_frame(calm_vals), [], panel, cfg, split=split, params=p)

    assert crash_r.objective.regime_independence is not None
    assert crash_r.objective.regime_independence > calm_r.objective.regime_independence
    # ...even though the crash specialist is the WEAKER factor overall (its edge is
    # confined to the 20% tail), which is exactly the point of a separate axis.
    assert crash_r.objective.marginal_value < calm_r.objective.marginal_value


# ── parsimony + sign consistency plumbing ─────────────────────────────────────

def test_parsimony_and_sign_consistency_are_wired():
    panel, rng = _panel()
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.002 * rng.standard_normal((N_BARS, len(TICKERS)))

    code = "def calc(self):\n    return (self.close - self.open) / self.high * 2"
    # a positive-IC factor with a declared +1 expected sign is sign-consistent
    r = evaluate_candidate(good, [], panel, cfg, candidate_code=code, expected_sign=1)
    assert r.objective.parsimony == -float(r.diagnostics["complexity"])
    assert r.diagnostics["complexity"] > 0
    assert r.diagnostics["sign_consistency"] is True

    # declaring the WRONG sign flags the inconsistency (a data-mining red flag)
    r_wrong = evaluate_candidate(good, [], panel, cfg, candidate_code=code, expected_sign=-1)
    assert r_wrong.diagnostics["sign_consistency"] is False


# ── split threading ───────────────────────────────────────────────────────────

def test_explicit_split_is_respected():
    panel, rng = _panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index, is_frac=0.5, val_frac=0.25)
    r = evaluate_candidate(_noise(rng), [], panel, cfg, split=split)
    assert r.raw["split_sizes"] == {"is": 300, "val": 150, "test": 150}
    # TEST is never scored: the CPCV folds live entirely within IS∪VAL
    assert r.diagnostics["cpcv_n_folds"] >= 2


def test_candidate_evaluation_ignores_test_rows_and_boundary_labels():
    panel, rng = _panel(seed=12)
    cfg = _cfg(target_horizon=12)
    split = three_way_split(panel["close"].index, is_frac=0.5, val_frac=0.25)
    params = EvalParams(cpcv_groups=4, cpcv_k=1)

    candidate = _noise(rng)
    book = [_noise(rng)]
    jitter = [_noise(rng)]

    clean = evaluate_candidate(
        candidate, book, panel, cfg, split=split,
        params=params, jitter_signals=jitter, candidate_id="leak_check",
    )

    poisoned_panel = {"close": _poison_test_rows(panel["close"], split)}
    poisoned = evaluate_candidate(
        _poison_test_rows(candidate, split),
        [_poison_test_rows(book[0], split)],
        poisoned_panel,
        cfg,
        split=split,
        params=params,
        jitter_signals=[_poison_test_rows(jitter[0], split)],
        candidate_id="leak_check",
    )

    _assert_nested_close(clean.to_dict(), poisoned.to_dict())


def test_set_evaluation_ignores_test_rows_and_boundary_labels():
    panel, rng = _panel(seed=21)
    cfg = _cfg(target_horizon=12)
    split = three_way_split(panel["close"].index, is_frac=0.5, val_frac=0.25)
    params = EvalParams(cpcv_groups=4, cpcv_k=1)

    members = {"a": _noise(rng), "b": _noise(rng)}
    clean = evaluate_set(
        members, panel, cfg, split=split, params=params, candidate_id="set_leak_check"
    )

    poisoned_members = {
        fid: _poison_test_rows(sig, split)
        for fid, sig in members.items()
    }
    poisoned = evaluate_set(
        poisoned_members,
        {"close": _poison_test_rows(panel["close"], split)},
        cfg,
        split=split,
        params=params,
        candidate_id="set_leak_check",
    )

    _assert_nested_close(clean.to_dict(), poisoned.to_dict())
