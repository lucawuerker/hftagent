"""Byte-equivalence of the seed executors vs the legacy hardcoded pipelines (E0).

The seeds ARE genome #0: if these tests fail, research and deployment have
drifted and the whole execution-evolution baseline is invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.positions import per_underlying_positions
from quant_fund_agent.backtesting.strategy_backtester import _signal_to_positions
from quant_fund_agent.execution.base import get_executor, run_executor, validate_weights
from quant_fund_agent.execution.state import build_state_frames

T, N = 300, 8


def _signal(seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=T, freq="D")
    cols = [f"S{i}" for i in range(N)]
    return pd.DataFrame(rng.normal(0, 1, (T, N)), index=idx, columns=cols)


def _close(seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-06-01", periods=T, freq="D")
    cols = [f"S{i}" for i in range(N)]
    return pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (T, N)), axis=0),
                        index=idx, columns=cols)


def test_topk_dollar_neutral_matches_legacy_pipeline():
    signal = _signal()
    cls = get_executor("topk_dollar_neutral")
    ex = cls()
    ex.overrides = {"holding_period": 3, "max_positions": 4,
                    "min_conviction": 0.5, "equal_weight": 0}
    got = ex.target_weights(signal, {})
    want = _signal_to_positions(signal, holding_period=3, max_positions=4,
                                equal_weight=False, min_conviction=0.5)
    pd.testing.assert_frame_equal(got, want)


def test_topk_default_params_match_legacy_defaults():
    signal = _signal(3)
    got = get_executor("topk_dollar_neutral")().target_weights(signal, {})
    want = _signal_to_positions(signal)  # library defaults: hp=1, K=20, eq=False, conv=0
    pd.testing.assert_frame_equal(got, want)


def test_zscore_threshold_matches_legacy_pipeline():
    signal = _signal(4)
    cls = get_executor("zscore_threshold_equal_weight")
    ex = cls()
    ex.overrides = {"n_max_positions": 5, "holding_period": 2,
                    "threshold": 0.8, "zscore_window": 100}
    got = ex.target_weights(signal, {})
    want = per_underlying_positions(signal, n_max_positions=5, holding_period=2,
                                    mode="threshold", threshold=0.8,
                                    zscore_basis="expanding", zscore_window=100)
    pd.testing.assert_frame_equal(got, want)


def test_seeds_satisfy_their_own_output_contract():
    signal, close = _signal(5), _close(6)
    state = build_state_frames({"close": close}, signal)
    for eid in ("topk_dollar_neutral", "zscore_threshold_equal_weight"):
        cls = get_executor(eid)
        w = run_executor(cls(), signal, state, close)
        assert validate_weights(w, cls.regime) == [], eid


def test_seeds_pass_through_run_executor_identically():
    """run_executor's vectorised path must not alter the seed output."""
    signal = _signal(7)
    close = _close(8)
    state = build_state_frames({"close": close}, signal)
    cls = get_executor("zscore_threshold_equal_weight")
    direct = cls().target_weights(signal, state)
    driven = run_executor(cls(), signal, state, close)
    pd.testing.assert_frame_equal(direct, driven)
