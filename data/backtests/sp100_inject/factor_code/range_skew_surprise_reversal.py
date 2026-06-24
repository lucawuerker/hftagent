"""Markets often overreact when a bar prints a large asymmetric excursion relative to its eventual close: a long lower tail that closes weakly, or a long upper tail that closes strongly, can indicate failed price discovery and trapped liquidity. Inspired by the paper's path-functional view of backward dynamics, this factor treats the intrabar path as information, not just the close, and looks for mean reversion after unusually one-sided range asymmetry that was not fully confirmed by the final close. The edge should be strongest when the extreme was large but the bar failed to sustain it, suggesting transient order-flow imbalance rather than genuine trend continuation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    abs_,
    correlation,
    delta,
    rank,
    stddev,
    ts_mean,
    ts_rank,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeSkewSurpriseReversal(BaseFactor):
    """Mean-reversion factor on unusually skewed intrabar range excursions."""

    factor_id = "range_skew_surprise_reversal"
    name = "Range Skew Surprise Reversal"
    category = "mean_reversion"
    description = (
        "Compute a signed intrabar skew score from the relative sizes of upper "
        "and lower wicks versus the real body, then compare it to recent "
        "cross-sectional normalization. The factor goes against unusually "
        "extreme skew when the close fails to validate the direction of the "
        "excursion."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-9
        body = abs_(close - open_)
        rng = (high - low).abs().replace(0, np.nan)
        upper = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower = (np.minimum(open_, close) - low).clip(lower=0.0)

        wick_asym = (lower - upper) / (rng + eps)
        body_pos = (close - open_) / (body + eps)
        skew_raw = wick_asym * (1.0 - 0.5 * body_pos.abs())

        extreme_skew = ts_rank(abs_(skew_raw.fillna(0.0)), 20)
        skew_z = skew_raw - ts_mean(skew_raw.fillna(0.0), 20)
        skew_vol = stddev(skew_raw.fillna(0.0), 20)
        normalized_skew = skew_z / (skew_vol + eps)

        close_strength = ts_rank((close - open_).fillna(0.0) / (rng.fillna(0.0) + eps), 20)
        validation_fail = 1.0 - close_strength

        reversal_pressure = -normalized_skew * (0.5 + 0.5 * extreme_skew) * (0.5 + 0.5 * validation_fail)

        surprise = delta(close, 1).fillna(0.0) / (ts_mean(rng.fillna(0.0), 20) + eps)
        surprise_rank = rank(surprise)
        context = correlation(skew_raw.fillna(0.0), close.fillna(0.0), 10)
        context = context.fillna(0.0)

        signal = reversal_pressure * (1.0 + 0.25 * surprise_rank.fillna(0.0)) * (1.0 - 0.25 * context.abs())
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
