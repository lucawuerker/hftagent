"""Tests for the N_trials-aware deflation helpers (P0 foundation)."""

from __future__ import annotations

import math

import pytest

from quant_fund_agent.research_eval.deflation import (
    deflated_ic,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    ic_haircut,
    _norm_cdf,
    _norm_ppf,
)


# ── normal helpers ────────────────────────────────────────────────────────────

def test_norm_helpers_roundtrip():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)
    for p in (0.01, 0.25, 0.75, 0.99):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-6)


# ── IC haircut ────────────────────────────────────────────────────────────────

def test_ic_haircut_zero_for_single_trial():
    assert ic_haircut(1000, 1) == 0.0
    assert ic_haircut(1, 100) == 0.0


def test_ic_haircut_grows_with_trials():
    h10 = ic_haircut(1000, 10)
    h1000 = ic_haircut(1000, 1000)
    assert 0 < h10 < h1000
    # closed form: sqrt(2 ln N / n_obs)
    assert h1000 == pytest.approx(math.sqrt(2 * math.log(1000) / 1000))


def test_deflated_ic_haircut_and_tstat():
    out = deflated_ic(best_abs_ic=0.05, n_obs=2000, n_trials=500)
    assert out["deflated_ic"] <= out["best_ic"]
    assert out["luck_ic"] > 0
    # more trials → a smaller deflated t-stat for the same raw IC
    fewer = deflated_ic(0.05, 2000, 5)
    assert fewer["deflated_t"] > out["deflated_t"]


def test_deflated_ic_single_trial_no_haircut():
    out = deflated_ic(0.05, 2000, 1)
    assert out["deflated_ic"] == pytest.approx(0.05, abs=1e-9)
    assert out["deflated_t"] == pytest.approx(0.05 * math.sqrt(2000), abs=1e-6)


# ── expected max Sharpe / deflated Sharpe ─────────────────────────────────────

def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(sr_variance=0.25, n_trials=1) == 0.0
    s10 = expected_max_sharpe(0.25, 10)
    s1000 = expected_max_sharpe(0.25, 1000)
    assert 0 < s10 < s1000


def test_deflated_sharpe_ratio_is_a_probability():
    p = deflated_sharpe_ratio(0.15, n_obs=1000, sr_variance=0.04, n_trials=100)
    assert p is not None and 0.0 <= p <= 1.0


def test_deflated_sharpe_higher_sr_more_significant():
    lo = deflated_sharpe_ratio(0.05, n_obs=1000, sr_variance=0.04, n_trials=100)
    hi = deflated_sharpe_ratio(0.25, n_obs=1000, sr_variance=0.04, n_trials=100)
    assert hi > lo


def test_deflated_sharpe_more_trials_less_significant():
    few = deflated_sharpe_ratio(0.15, n_obs=1000, sr_variance=0.04, n_trials=10)
    many = deflated_sharpe_ratio(0.15, n_obs=1000, sr_variance=0.04, n_trials=5000)
    assert few > many


def test_deflated_sharpe_needs_threshold_inputs():
    with pytest.raises(ValueError, match="sr_star"):
        deflated_sharpe_ratio(0.15, n_obs=1000)
    # an explicit sr_star also works
    p = deflated_sharpe_ratio(0.15, n_obs=1000, sr_star=0.1)
    assert p is not None and 0.0 <= p <= 1.0
