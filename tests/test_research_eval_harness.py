"""Tests for the deterministic fitness harness (P0 foundation).

All on a small synthetic panel — no LOBSTER / market data required.  The harness is
signal-based, so we feed it factor signal frames directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval import harness as harness_mod
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


def test_deflation_is_a_diagnostic_not_a_search_gate():
    """WS1: N_trials deflation is no longer a per-candidate *search* gate — it is a
    diagnostic here (``deflation_ok is None``) and is enforced once, at publish time
    (``research_eval.publish``).  A lucky-noise factor therefore stays *search*-
    selectable (the discovery half of the two modes), but its deflated t-stat under a
    large trial count is not positive, so the publish filter will drop it."""
    panel, rng = _panel()
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.002 * rng.standard_normal((N_BARS, len(TICKERS)))
    noise = _noise(rng)

    params = EvalParams(n_trials=5000)
    good_r = evaluate_candidate(good, [], panel, cfg, params=params)
    noise_r = evaluate_candidate(noise, [], panel, cfg, params=params)

    # deflation never gates search eligibility any more
    assert good_r.gates.deflation_ok is None
    assert noise_r.gates.deflation_ok is None
    # but the deflated t-stat is still computed (teacher channel) and separates them
    assert good_r.diagnostics["deflation"]["deflated_t"] > 0
    assert (good_r.diagnostics["deflation"]["deflated_t"]
            > noise_r.diagnostics["deflation"]["deflated_t"])
    # the predictive factor still clears the *search* gates (coverage + degradation)
    assert good_r.selectable


# ── P5 economic reward (WS3): cost gate + perturbation robustness probe ────────

def test_cost_gate_is_off_by_default_and_rejects_high_turnover_when_enabled():
    panel, rng = _panel(seed=3)
    cfg = _cfg()
    fast = _frame(rng.standard_normal((N_BARS, len(TICKERS))))  # fast, high-turnover
    # off by default → cost_ok not evaluated, but turnover is still a diagnostic
    r_off = evaluate_candidate(fast, [], panel, cfg, params=EvalParams())
    assert r_off.gates.cost_ok is None
    assert r_off.diagnostics["turnover"] is not None
    assert r_off.diagnostics["net_ret"] is not None
    # enabled with a strict floor → the high-turnover factor is rejected
    r_on = evaluate_candidate(fast, [], panel, cfg,
                              params=EvalParams(gate_turnover=0.01))
    assert r_on.gates.cost_ok is False
    assert not r_on.selectable


def test_perturbation_probe_only_lowers_robustness_when_enabled():
    panel, rng = _panel(seed=4)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.01 * rng.standard_normal((N_BARS, len(TICKERS)))
    base = evaluate_candidate(good, [], panel, cfg, params=EvalParams())
    pert = evaluate_candidate(
        good, [], panel, cfg,
        params=EvalParams(perturbation_weight=5.0, perturbation_sigma=1.0))
    # the probe is a penalty: it can only lower (or leave) robustness, never raise it
    assert pert.objective.robustness <= base.objective.robustness + 1e-9
    assert pert.diagnostics["perturbation_penalty"] is not None
    assert pert.diagnostics["perturbation_penalty"] >= 0.0
    # default run leaves the baseline arm untouched (no perturbation term)
    assert base.diagnostics["perturbation_penalty"] is None


# ── WS4: factor-zoo dedup novelty diagnostic (DIAG only, no gate) ──────────────

def test_zoo_dedup_flags_a_rediscovered_factor():
    panel, rng = _panel(seed=5)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    known = fwd + 0.01 * rng.standard_normal((N_BARS, len(TICKERS)))
    near_copy = known + 0.001 * rng.standard_normal((N_BARS, len(TICKERS)))
    novel = _frame(rng.standard_normal((N_BARS, len(TICKERS))))  # orthogonal to `known`

    ref_ids = ["known_alpha"]
    # a near-copy of a reference factor → high max-|corr| (rediscovery flag)
    r_copy = evaluate_candidate(near_copy, [], panel, cfg,
                                reference_signals=[known], reference_ids=ref_ids,
                                reference_codes=["x = close.pct_change()"],
                                candidate_code="x = close.pct_change()")
    assert r_copy.diagnostics["zoo_max_abs_corr"] > 0.8
    assert r_copy.diagnostics["zoo_nearest"] == "known_alpha"
    assert r_copy.diagnostics["zoo_min_code_distance"] == 0.0  # identical source
    # a genuinely novel factor → low correlation to the reference
    r_novel = evaluate_candidate(novel, [], panel, cfg,
                                 reference_signals=[known], reference_ids=ref_ids,
                                 candidate_code="y = volume.rolling(20).std()",
                                 reference_codes=["x = close.pct_change()"])
    assert r_novel.diagnostics["zoo_max_abs_corr"] < 0.3
    # it is a DIAGNOSTIC, never a gate — novelty does not change selectability
    assert r_copy.gates.to_dict()["passed"] == r_copy.selectable
    # default (no reference) leaves the keys None
    r_off = evaluate_candidate(novel, [], panel, cfg)
    assert r_off.diagnostics["zoo_max_abs_corr"] is None


# ── WS2: QD behavior descriptors (computed, but NOT scored) ────────────────────

def test_behavior_descriptors_present_and_separate_from_objective():
    from quant_fund_agent.research_eval.fitness import ObjectiveVector

    panel, rng = _panel(seed=6)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    sig = fwd + 0.01 * rng.standard_normal((N_BARS, len(TICKERS)))
    r = evaluate_candidate(sig, [], panel, cfg)
    # behavior descriptors are produced …
    assert set(r.behavior) >= {"trend_reversal", "signal_speed", "stress_activation"}
    assert r.behavior["signal_speed"] is not None
    # … but they are NOT part of the scored objective vector (still exactly 5 axes)
    assert ObjectiveVector.AXES == (
        "marginal_value", "independence", "robustness", "parsimony",
        "regime_independence")
    assert "trend_reversal" not in ObjectiveVector.AXES
    # round-trips through the MCP/state serialisation
    assert r.to_dict()["behavior"]["signal_speed"] == r.behavior["signal_speed"]


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


def test_cpcv_robustness_refits_marginal_model_per_fold(monkeypatch):
    """The scored robustness axis is fold-refit LOCO, not raw-signal CPCV."""
    panel, f1, f2, _ = _two_driver_panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index, is_frac=0.6, val_frac=0.2)
    params = EvalParams(cpcv_groups=4, cpcv_k=1, cpcv_model="ridge",
                        cpcv_fast=False)
    calls: list[tuple[int, tuple[int, ...]]] = []
    original = harness_mod._combined_prediction

    def spy(signals, close, is_mask, cfg, model):
        calls.append((len(signals), tuple(np.flatnonzero(is_mask))))
        return original(signals, close, is_mask, cfg, model)

    monkeypatch.setattr(harness_mod, "_combined_prediction", spy)
    r = evaluate_candidate(f2, [f1], panel, cfg, split=split,
                           params=params, candidate_id="fresh")

    fold_trains = {
        mask for n_signals, mask in calls
        if n_signals == 2 and mask != tuple(np.flatnonzero(split.is_mask))
    }
    assert r.diagnostics["cpcv_score_kind"] == "refit_marginal_delta"
    assert r.diagnostics["cpcv_model"] == "ridge"
    assert r.diagnostics["cpcv_n_folds"] >= 2
    assert r.diagnostics["standalone_cpcv_n_folds"] >= 2
    assert len(fold_trains) >= 2


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
