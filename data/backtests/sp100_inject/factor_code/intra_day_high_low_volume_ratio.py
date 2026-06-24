"""Intra-day price extremes often signal potential reversals or continuations, particularly when accompanied by high volume. By examining the ratio of volume during high and low price movements within a trading session, we can gauge the strength of the price action and the likelihood of follow-through. This concept aligns with market structure theories that suggest volume acts as a confirmation of price moves."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_max, ts_min, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntraDayHighLowVolumeRatio(BaseFactor):
    factor_id = "intra_day_high_low_volume_ratio"
    name = "Intra-Day High-Low Volume Ratio"
    category = "momentum"
    inputs = ["high", "low", "volume"]
    description = "This factor calculates the ratio of volume traded when the price hits its intra-day high to the volume when it hits its intra-day low, providing insights on market conviction behind price extremes."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Calculate the volume when hitting intra-day high and low
        high_volume = volume.where(high == high.rolling(window=1).max(), 0).fillna(0.0)
        low_volume = volume.where(low == low.rolling(window=1).min(), 0).fillna(0.0)

        # Calculate the total volume for high and low
        total_high_volume = ts_sum(high_volume, 1)
        total_low_volume = ts_sum(low_volume, 1)

        # Avoid division by zero
        total_low_volume = total_low_volume.replace(0, np.nan)

        # Calculate the ratio
        ratio = total_high_volume / total_low_volume

        return ratio