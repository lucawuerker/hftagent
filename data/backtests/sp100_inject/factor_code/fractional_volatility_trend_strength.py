"""Building on the insights from the fractional volatility model, this factor captures the persistence of volatility trends. As demonstrated in the research, volatility exhibits long-range dependence and can be influenced by fractional noise, indicating that recent volatility behavior can predict future movements. By analyzing the strength and direction of these trends, traders can identify potential periods of price acceleration or deceleration."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, stddev, ts_max, ts_argmax, ts_rank, returns, vwap)

@register_factor
class FractionalVolatilityTrendStrength(BaseFactor):
    factor_id = "fractional_volatility_trend_strength"
    name = "Fractional Volatility Trend Strength"
    category = "volatility"
    inputs = ["high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)

        # Calculate the volatility as the difference between high and low prices
        volatility = high - low

        # Calculate the trend strength using the sum of recent volatility changes
        trend_strength = ts_sum(volatility, 5)

        # Normalize the trend strength
        trend_strength_normalized = trend_strength / stddev(trend_strength, 5)

        # Return the trend strength with the same index and columns as the close prices
        return trend_strength_normalized.where(trend_strength_normalized.notna(), 0.0)