"""Tests for end-of-run book curation (Lever 2): greedy + elastic-net."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.comparison.config import ComparisonConfig
from quant_fund_agent.research_eval import curation
from quant_fund_agent.research_eval.splits import three_way_split

N = 600
TICKERS = ["A", "B", "C", "D", "E"]


def _panel(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=N, freq="min")
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.standard_normal((N, len(TICKERS))) * 0.01, axis=0),
        index=idx, columns=TICKERS)
    return close, idx, rng


def _cfg():
    return ComparisonConfig(preruns=["z"], target_horizon=1,
                            fit_standardize="per_underlying", seed=0)


def _frame(values, idx):
    return pd.DataFrame(values, index=idx, columns=TICKERS)


def test_greedy_keeps_strong_drops_noise_and_auto_sizes():
    close, idx, rng = _panel()
    fwd = close.pct_change().shift(-1).fillna(0.0)
    # Noise is scaled by the target's own volatility so the tiers are genuinely
    # distinct: unscaled N(0,1) noise dwarfs a ~1% return and would make
    # "strong"'s true IC (~0.03) statistically indistinguishable from pure
    # noise's sampling error on the VAL window.
    scale = float(fwd.std().mean())
    signals = {
        "strong": fwd + 0.3 * scale * _frame(rng.standard_normal((N, len(TICKERS))), idx),
        "medium": fwd + 3.0 * scale * _frame(rng.standard_normal((N, len(TICKERS))), idx),
        "noise": _frame(rng.standard_normal((N, len(TICKERS))), idx),
    }
    split = three_way_split(idx, is_frac=0.5, val_frac=0.25)
    out = curation.curate("greedy", signals, close, _cfg(), split)

    assert out["mode"] == "greedy"
    assert "strong" in out["kept_factor_ids"]
    assert "noise" not in out["kept_factor_ids"]        # never lifts the combined IC
    assert 1 <= len(out["kept_factor_ids"]) < 3         # auto-sized, not "keep all"


def test_greedy_respects_n_keep():
    close, idx, rng = _panel(1)
    fwd = close.pct_change().shift(-1).fillna(0.0)
    scale = float(fwd.std().mean())
    signals = {f"f{i}": fwd + (0.3 + 0.2 * i) * scale * _frame(
        rng.standard_normal((N, len(TICKERS))), idx) for i in range(4)}
    split = three_way_split(idx, is_frac=0.5, val_frac=0.25)
    out = curation.curate("greedy", signals, close, _cfg(), split, n_keep=2)
    assert len(out["kept_factor_ids"]) == 2


def test_elastic_net_returns_ranked_subset():
    close, idx, rng = _panel(2)
    fwd = close.pct_change().shift(-1).fillna(0.0)
    scale = float(fwd.std().mean())
    signals = {
        "strong": fwd + 0.2 * scale * _frame(rng.standard_normal((N, len(TICKERS))), idx),
        "weak": fwd + 2.0 * scale * _frame(rng.standard_normal((N, len(TICKERS))), idx),
        "noise": _frame(rng.standard_normal((N, len(TICKERS))), idx),
    }
    split = three_way_split(idx, is_frac=0.5, val_frac=0.25)
    out = curation.curate("elastic_net", signals, close, _cfg(), split,
                          n_keep=2, n_resamples=8)
    assert out["mode"] == "elastic_net"
    assert len(out["kept_factor_ids"]) == 2
    assert set(out["kept_factor_ids"]) <= set(signals)
    assert set(out["selection_frequency"]) == set(signals)


def test_singleton_pool_is_kept():
    close, idx, _ = _panel(3)
    split = three_way_split(idx, is_frac=0.5, val_frac=0.25)
    signals = {"only": close.pct_change().fillna(0.0)}
    for mode in ("greedy", "elastic_net"):
        out = curation.curate(mode, signals, close, _cfg(), split)
        assert out["kept_factor_ids"] == ["only"]
