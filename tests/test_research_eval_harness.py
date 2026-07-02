"""Tests for the deterministic fitness harness (P0 foundation).

All on a small synthetic panel — no LOBSTER / market data required.  The harness is
signal-based, so we feed it factor signal frames directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval.harness import EvalParams, evaluate_candidate
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


# ── independence axis ─────────────────────────────────────────────────────────

def test_independence_penalises_redundant_candidate():
    panel, rng = _panel()
    cfg = _cfg()
    base = _noise(rng)
    book = [base]
    duplicate = base + 1e-6 * rng.standard_normal((N_BARS, len(TICKERS)))  # ~ copy of book[0]
    fresh = _noise(rng)                                                     # unrelated

    dup_r = evaluate_candidate(duplicate, book, panel, cfg, candidate_id="dup")
    fresh_r = evaluate_candidate(fresh, book, panel, cfg, candidate_id="fresh")

    assert dup_r.diagnostics["max_abs_corr"] > 0.9
    assert fresh_r.diagnostics["max_abs_corr"] < 0.3
    # the fresh factor raises independence more than the near-duplicate
    assert fresh_r.objective.independence > dup_r.objective.independence


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
