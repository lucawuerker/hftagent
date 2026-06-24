"""Compute a cross-sectional score from the change in close location within the daily bar range, scaled by the bar range and recent range stability. Higher values indicate a stronger rotation of price settlement toward one end of the bar after accounting for how compressed or expanded the range is."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PolarRangeRotationMisalignment(BaseFactor):
    """Rotation of close location within range adjusted for range stability."""

    factor_id = "polar_range_rotation_misalignment"
    name = "Polar Range Rotation Misalignment"
    category = "other"
    inputs = ["open", "high", "low", "close"]
    window_length = 20
    description = (
        "Compute a cross-sectional score from the change in close location "
        "within the daily bar range, scaled by the bar range and recent range "
        "stability. Higher values indicate a stronger rotation of price "
        "settlement toward one end of the bar after accounting for how "
        "compressed or expanded the range is."
    )

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        open_loc = ((open_ - low) / bar_range).clip(lower=0.0, upper=1.0)
        close_loc = ((close - low) / bar_range).clip(lower=0.0, upper=1.0)

        rotation = close_loc - open_loc
        signed_direction = np.sign(close - open_)
        range_magnitude = abs_(close - open_) / bar_range

        range_stability = ts_mean(bar_range, 10) / (stddev(bar_range, 10) + 1e-12)
        stability = range_stability.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        intraday_alignment = rotation * signed_direction
        core = intraday_alignment * range_magnitude * stability

        cross_sectional = rank(core)
        return cross_sectional.fillna(0.0)
