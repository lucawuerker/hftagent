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

    params = EvalParams(n_trials=1, stability_blocks=4)
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
    # The predictive factor clears the search gates; temporal degradation remains
    # available only as a teacher diagnostic.
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


def test_perturbation_probe_only_lowers_marginal_value_when_enabled():
    panel, rng = _panel(seed=4)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.01 * rng.standard_normal((N_BARS, len(TICKERS)))
    base = evaluate_candidate(good, [], panel, cfg, params=EvalParams())
    pert = evaluate_candidate(
        good, [], panel, cfg,
        params=EvalParams(perturbation_weight=5.0, perturbation_sigma=1.0))
    # the probe is a penalty folded onto the marginal-value axis: the axis equals the
    # raw ΔIC minus perturbation_weight·penalty, so it can only lower (or leave) it
    assert pert.diagnostics["perturbation_penalty"] is not None
    assert pert.diagnostics["perturbation_penalty"] >= 0.0
    assert pert.objective.marginal_value == pytest.approx(
        pert.diagnostics["marginal_value_raw"]
        - 5.0 * pert.diagnostics["perturbation_penalty"])
    assert pert.objective.marginal_value <= pert.diagnostics["marginal_value_raw"] + 1e-9
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


def test_residual_ic_independence_scored_on_is_union_val():
    """Change 2: the residual-IC axis scores on IS∪VAL (betas still fit on IS), so an
    edge that lives in the IS era is visible where the old VAL-only window saw ~0."""
    from quant_fund_agent.research_eval.harness import _residual_ic

    rng = np.random.default_rng(3)
    n_cols = len(TICKERS)
    idx = pd.date_range("2022-01-01", periods=N_BARS, freq="D")
    split = three_way_split(idx, is_frac=0.6, val_frac=0.2)
    is_rows = np.asarray(split.is_mask)

    book_drv = rng.standard_normal((N_BARS, n_cols))        # the book's driver
    cand_drv = rng.standard_normal((N_BARS, n_cols))        # orthogonal to the book
    # forward return = book driver everywhere + candidate driver ONLY on IS bars
    fwd = book_drv.copy()
    fwd[is_rows] += 1.5 * cand_drv[is_rows]
    ret = np.zeros((N_BARS, n_cols)); ret[1:] = 0.01 * fwd[:-1]
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=idx, columns=TICKERS)
    panel = {"close": close}
    cfg = _cfg()
    book = [pd.DataFrame(book_drv, index=idx, columns=TICKERS)]
    cand = pd.DataFrame(cand_drv, index=idx, columns=TICKERS)

    ic_val = _residual_ic(cand, book, panel, cfg, split.is_mask, split.val_mask,
                          split.is_val_mask)                 # the OLD VAL-only window
    ic_dev = _residual_ic(cand, book, panel, cfg, split.is_mask, split.is_val_mask,
                          split.is_val_mask)                 # the NEW IS∪VAL window
    assert abs(ic_val) < 0.05                                # invisible on VAL-only
    assert abs(ic_dev) > 0.1                                 # material on IS∪VAL
    # the harness wires the independence axis to the IS∪VAL window
    r = evaluate_candidate(cand, book, panel, cfg, split=split)
    assert r.diagnostics["residual_ic"] == pytest.approx(ic_dev)
    assert r.objective.independence == pytest.approx(ic_dev)


# ── structural novelty axis (AST-based originality) ──────────────────────────

def test_structural_novelty_rewards_novel_code():
    """A factor with structurally different code scores higher novelty than a clone."""
    panel, rng = _panel(seed=5)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    sig = _frame(fwd.to_numpy(dtype=float) + 0.01 * rng.standard_normal((N_BARS, len(TICKERS))))

    book_code = (
        "def calc(self):\n"
        "    return (self.close - self.close.shift(5)) / self.close.shift(5)"
    )
    # structural clone: same pattern, only the window constant changed
    clone_code = (
        "def calc(self):\n"
        "    return (self.close - self.close.shift(10)) / self.close.shift(10)"
    )
    # genuinely novel: different fields, different operator structure
    novel_code = (
        "def calc(self):\n"
        "    vol = self.volume.rolling(20).mean()\n"
        "    return (vol - vol.shift(1)) / (vol.shift(1) + 1e-8)"
    )

    clone_r = evaluate_candidate(sig, [], panel, cfg,
                                 candidate_code=clone_code,
                                 book_codes=[book_code])
    novel_r = evaluate_candidate(sig, [], panel, cfg,
                                 candidate_code=novel_code,
                                 book_codes=[book_code])

    # Both have non-None structural_novelty (book is non-empty)
    assert clone_r.objective.structural_novelty is not None
    assert novel_r.objective.structural_novelty is not None
    # The novel factor is structurally further from the book than the clone
    assert novel_r.objective.structural_novelty > clone_r.objective.structural_novelty
    # Diagnostic keys are present
    assert "novelty_min_book_distance" in clone_r.diagnostics
    assert "novelty_min_zoo_distance" in clone_r.diagnostics


def test_structural_novelty_none_when_no_code():
    """Structural novelty is None (unmeasured) when no candidate code is provided."""
    panel, rng = _panel(seed=6)
    cfg = _cfg()
    r = evaluate_candidate(_noise(rng), [], panel, cfg)
    assert r.objective.structural_novelty is None


def test_structural_novelty_falls_back_to_zoo():
    """When book is empty, structural novelty falls back to the zoo reference."""
    panel, rng = _panel(seed=7)
    cfg = _cfg()
    code = "def calc(self):\n    return self.close.pct_change()"
    zoo_code = "def calc(self):\n    return self.close.pct_change()"  # identical
    # With an identical zoo reference, novelty should be near 0
    r = evaluate_candidate(_noise(rng), [], panel, cfg,
                           candidate_code=code,
                           book_codes=[],
                           reference_codes=[zoo_code])
    assert r.objective.structural_novelty is not None
    assert r.objective.structural_novelty < 0.1  # near-zero distance → low novelty


def test_structural_novelty_is_ast_based_not_character_edit():
    """A pure window-constant change is a canonical-AST clone (distance 0),
    which a character-edit proxy would have scored as *different* code."""
    panel, rng = _panel(seed=8)
    cfg = _cfg()
    book_code = ("def calc(self):\n"
                 "    return (self.close - self.close.shift(5)) / self.close.shift(5)")
    window_only = ("def calc(self):\n"
                   "    return (self.close - self.close.shift(200)) / self.close.shift(200)")
    r = evaluate_candidate(_noise(rng), [], panel, cfg,
                           candidate_code=window_only, book_codes=[book_code])
    assert r.objective.structural_novelty == 0.0
    assert r.diagnostics["novelty_metric"] == "canonical_ast_weighted_subtree_jaccard"
    assert r.diagnostics["novelty_nearest_book_similarity"] == 1.0
    assert r.diagnostics["novelty_candidate_ast_nodes"] > 0
    assert r.diagnostics["novelty_candidate_unique_subtrees"] > 0


def test_structural_novelty_nearest_book_idx_survives_skipped_codes():
    """The nearest-book index must refer to the ORIGINAL book position even when
    earlier book codes are empty/invalid and skipped (regression for the fix)."""
    panel, rng = _panel(seed=9)
    cfg = _cfg()
    candidate = "def calc(self):\n    return self.volume.rolling(20).std()"
    # index 0 and 1 are unmeasurable; index 2 is the true structural clone
    book_codes = ["", "not valid python (",
                  "def calc(self):\n    return self.volume.rolling(50).std()"]
    r = evaluate_candidate(_noise(rng), [], panel, cfg,
                           candidate_code=candidate, book_codes=book_codes)
    assert r.diagnostics["novelty_nearest_book_idx"] == 2      # not 0
    assert r.diagnostics["novelty_min_book_distance"] == 0.0   # window-only clone


def test_set_structural_novelty_mean_nearest_neighbour_and_singleton():
    """SET-mode structural novelty = mean nearest-neighbour canonical-AST distance;
    a singleton set scores 1.0; no measurable member codes → None."""
    panel, rng = _panel(seed=10)
    cfg = _cfg()
    members = {"a": _noise(rng), "b": _noise(rng), "c": _noise(rng)}
    # a and b are canonical clones (window only); c is structurally different
    codes = {
        "a": "def calc(self):\n    return self.close - self.close.shift(5)",
        "b": "def calc(self):\n    return self.close - self.close.shift(9)",
        "c": "def calc(self):\n    return self.volume.rolling(20).mean()",
    }
    r = evaluate_set(members, panel, cfg, member_codes=codes, candidate_id="set")
    nov = r.objective.structural_novelty
    assert nov is not None and 0.0 < nov < 1.0
    # a↔b nearest-neighbour is 0 (clones); c's nearest is > 0 → mean strictly positive
    assert r.diagnostics["structural_novelty"] == nov

    single = evaluate_set({"a": members["a"]}, panel, cfg,
                          member_codes={"a": codes["a"]}, candidate_id="one")
    assert single.objective.structural_novelty == 1.0

    no_codes = evaluate_set(members, panel, cfg, candidate_id="none")
    assert no_codes.objective.structural_novelty is None


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


def test_marginal_penalties_fold_onto_axis_without_refit(monkeypatch):
    """The plateau/perturbation/sign penalties fold onto the marginal-value axis and
    never refit a model — only the with-book and book-only marginal fits happen."""
    panel, f1, f2, _ = _two_driver_panel()
    cfg = _cfg()
    split = three_way_split(panel["close"].index, is_frac=0.6, val_frac=0.2)
    params = EvalParams(marginal_model="ridge")
    calls: list[tuple[int, tuple[int, ...]]] = []
    original = harness_mod._combined_prediction

    def spy(signals, close, is_mask, cfg, model):
        calls.append((len(signals), tuple(np.flatnonzero(is_mask))))
        return original(signals, close, is_mask, cfg, model)

    monkeypatch.setattr(harness_mod, "_combined_prediction", spy)
    r = evaluate_candidate(f2, [f1], panel, cfg, split=split,
                           params=params, candidate_id="fresh")

    # exactly the with-book and book-only fits from _marginal_value; the penalties
    # fold onto the axis and fit nothing extra.
    assert calls == [
        (2, tuple(np.flatnonzero(split.is_mask))),
        (1, tuple(np.flatnonzero(split.is_mask))),
    ]
    # the robustness axis is gone
    assert not hasattr(r.objective, "robustness")
    assert "robustness" not in r.objective.to_dict()
    # with no jitter, perturbation off and no expected_sign, the axis equals the raw ΔIC
    assert r.diagnostics["marginal_value_raw"] is not None
    assert r.objective.marginal_value == pytest.approx(
        r.diagnostics["marginal_value_raw"])


# ── temporal degradation is diagnostic-only ──

def test_temporal_degradation_is_diagnostic_not_axis_or_gate():
    """A sign-flipped VAL edge passes and reports degradation to the teacher."""
    panel, rng = _panel(seed=3)
    cfg = _cfg()
    split = three_way_split(panel["close"].index, is_frac=0.6, val_frac=0.2)
    fwd = panel["close"].pct_change().shift(-1)

    # predictive on IS, sign-reversed on VAL → is_ic > 0, val_ic < 0
    sig = fwd.copy()
    val_rows = np.flatnonzero(split.val_mask)
    sig.iloc[val_rows] = -fwd.iloc[val_rows]
    sig = sig + 0.001 * _frame(rng.standard_normal((N_BARS, len(TICKERS))))

    r = evaluate_candidate(sig, [], panel, cfg, split=split)
    assert r.selectable
    assert not hasattr(r.objective, "temporal_robustness")
    assert r.diagnostics["degradation_ratio"] < 0


def test_temporal_degradation_none_below_is_floor_leaves_gates_intact():
    panel, rng = _panel(seed=4)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    sig = fwd + 0.01 * _frame(rng.standard_normal((N_BARS, len(TICKERS))))
    r = evaluate_candidate(sig, [], panel, cfg, params=EvalParams(min_is_ic=10.0))
    assert r.diagnostics["degradation_ratio"] is None
    assert r.gates.coverage_ok  # gates do not depend on the axis


def test_plateau_penalty_folds_onto_marginal_value():
    """A window-jitter plateau penalty is subtracted from the marginal-value axis."""
    panel, rng = _panel(seed=7)
    cfg = _cfg()
    fwd = panel["close"].pct_change().shift(-1)
    good = fwd + 0.01 * rng.standard_normal((N_BARS, len(TICKERS)))
    # a jitter variant that is pure noise → its VAL IC collapses → a plateau penalty
    jitter = [_frame(rng.standard_normal((N_BARS, len(TICKERS))))]
    r = evaluate_candidate(good, [], panel, cfg, jitter_signals=jitter,
                           params=EvalParams(plateau_weight=1.0))
    assert r.diagnostics["plateau_penalty"] is not None
    assert r.diagnostics["plateau_penalty"] > 0.0
    assert r.objective.marginal_value == pytest.approx(
        r.diagnostics["marginal_value_raw"] - r.diagnostics["plateau_penalty"])


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
    # TEST is never scored: the fitness is measured on VAL rows only.
    assert r.diagnostics["val_n_obs"] > 0


def test_candidate_evaluation_ignores_test_rows_and_boundary_labels():
    panel, rng = _panel(seed=12)
    cfg = _cfg(target_horizon=12)
    split = three_way_split(panel["close"].index, is_frac=0.5, val_frac=0.25)
    params = EvalParams(stability_blocks=4)

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
    params = EvalParams(stability_blocks=4)

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


# ── J3: executor-aware cost gate (coupling) ────────────────────────────────────

SOTA_EXEC_CODE = '''\
"""Coupling-test executor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.execution.base import BaseExecutor, register_executor


@register_executor
class CouplingTestExec(BaseExecutor):
    executor_id = "coupling_test_exec"
    name = "sign 1/n"
    regime = "per_underlying"
    inputs = ["signal"]
    params = {"n_max": 5.0}

    def target_weights(self, signal, state):
        return np.sign(signal).fillna(0.0) / 5.0
'''


def test_cost_executor_none_is_byte_identical():
    """THE load-bearing J3 regression: coupling off == pre-J3 behaviour."""
    panel, rng = _panel()
    candidate = _noise(rng)
    split = three_way_split(panel["close"].index)

    base = evaluate_candidate(candidate, [], panel, _cfg(), split,
                              params=EvalParams(), candidate_id="c")
    explicit = evaluate_candidate(candidate, [], panel, _cfg(), split,
                                  params=EvalParams(cost_executor=None),
                                  candidate_id="c")
    _assert_nested_close(base.to_dict(), explicit.to_dict())
    assert "net_capture_sota" not in base.diagnostics


def test_cost_executor_gates_and_diagnoses():
    panel, rng = _panel()
    candidate = _noise(rng)
    split = three_way_split(panel["close"].index)
    sota = {"executor_id": "coupling_test_exec", "code": SOTA_EXEC_CODE}

    res = evaluate_candidate(candidate, [], panel, _cfg(), split,
                             params=EvalParams(cost_executor=sota),
                             candidate_id="c")
    # the coupling diagnostics are present and the gate was EVALUATED
    assert "net_capture_sota" in res.diagnostics
    assert res.diagnostics["cost_executor_id"] == "coupling_test_exec"
    assert res.gates.cost_ok is not None
    # a pure-noise candidate traded through an executor at 5 bps: net ≤ 0
    if res.gates.cost_ok is False:
        assert "cost_executor" in res.gates.reasons


def test_cost_executor_failure_falls_back_not_crashes():
    panel, rng = _panel()
    candidate = _noise(rng)
    split = three_way_split(panel["close"].index)
    broken = {"executor_id": "broken_exec", "code": "this is not python ("}

    res = evaluate_candidate(candidate, [], panel, _cfg(), split,
                             params=EvalParams(cost_executor=broken),
                             candidate_id="c")
    # fell back to the explicit construction; run completed; diagnostics stamped
    assert res.diagnostics["turnover"] is not None
    assert "net_capture_sota" in res.diagnostics
