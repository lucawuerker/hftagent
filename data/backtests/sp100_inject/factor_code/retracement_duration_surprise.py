"""The survey shows that trend retracements and their durations are positively skewed, roughly log-normal, and that longer-than-typical corrections tend to accompany larger delays in trend recognition. That suggests a useful signal is not the raw pullback itself, but how unusually long the current bar’s local retracement is relative to its own recent history: an unusually persistent pullback after a strong advance often reflects trend exhaustion, while an unusually quick retracement can indicate resilient trend continuation. This is a structural, path-dependent measure inspired by the paper’s emphasis on log-normal trend variables and delay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import rank, stddev, ts_mean, ts_sum


@register_factor
class RetracementDurationSurprise(BaseFactor):
    """Cross-sectional surprise in log retracement duration relative to history."""

    factor_id = "retracement_duration_surprise"
    name = "Retracement Duration Surprise"
    category = "other"
    description = (
        "Measure each ticker’s recent normalized drawdown length over a rolling "
        "window using close-to-close, open, high, and low, then compare it to a "
        "slower baseline of its own history. Output a cross-sectional z-score of "
        "the log retracement duration ratio, with positive values indicating "
        "unusually persistent retracements and negative values indicating "
        "unusually brief pullbacks."
    )
    window_length = 60
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        bar_range = (high - low).abs()
        up_move = (close - prev_close).clip(lower=0.0)
        down_move = (prev_close - close).clip(lower=0.0)

        # Proxy for how many bars a retracement has been persisting:
        # longer persistence is inferred when downside pressure dominates
        # while realized range stays compressed.
        raw_pressure = down_move / (bar_range + 1e-12)
        raw_pressure = raw_pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Local retracement duration proxy over a medium window.
        dur_proxy = ts_sum(raw_pressure, 10)

        # Normalize by a slower baseline of its own history.
        baseline = ts_mean(dur_proxy, 40)
        dispersion = stddev(dur_proxy, 40)
        log_ratio = np.log((dur_proxy + 1e-12) / (baseline + 1e-12))
        log_ratio = log_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        surprise = (log_ratio - ts_mean(log_ratio, 40)) / (dispersion + 1e-12)
        surprise = surprise.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional z-score proxy using ranks; centered around zero.
        cs = (rank(surprise) - 0.5) * 2.0
        cs = cs.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Slightly emphasize unusually persistent pullbacks after strong advances.
        advance_strength = ts_sum(up_move / (bar_range + 1e-12), 10)
        advance_strength = advance_strength.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        out = cs * (1.0 + 0.25 * advance_strength)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out
