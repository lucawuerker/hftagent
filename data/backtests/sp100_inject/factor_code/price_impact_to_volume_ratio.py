"""This factor aims to quantify the efficiency of trading by measuring the relationship between price impact and trading volume. According to the insights from the stochastic thermodynamics framework, a higher ratio of price impact relative to volume could indicate less efficient execution, leading to potential mispricing opportunities that can be exploited. As traders seek to minimize their price impact, those with higher ratios could signal overreaction or underreaction in price adjustments."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceImpactToVolumeRatio(BaseFactor):
    factor_id = "price_impact_to_volume_ratio"
    name = "Price Impact to Volume Ratio"
    category = "microstructure"
    description = ("This signal computes the ratio of the price impact (using the difference between open and close prices) to the volume traded over a specific period. A higher ratio may indicate increased inefficiency in market pricing.")
    inputs = ["open", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_price = data["open"].fillna(0.0)
        close_price = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        price_impact = delta(close_price, 1)  # Price impact as the change from open to close
        price_impact_sum = ts_sum(price_impact, 5)  # Sum of price impact over the last 5 bars
        volume_sum = ts_sum(volume, 5)  # Sum of volume over the last 5 bars

        # Avoid division by zero by replacing zeros in volume_sum with NaN
        volume_sum = volume_sum.replace(0, np.nan)

        ratio = price_impact_sum / volume_sum  # Calculate the ratio
        return ratio