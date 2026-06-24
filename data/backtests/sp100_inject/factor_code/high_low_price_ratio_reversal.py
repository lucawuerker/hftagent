"""The high-low price ratio captures the relative strength of a stock's price movement within a given period. When this ratio reaches extremes, it can indicate overbought or oversold conditions, suggesting a potential reversal in price direction. This behavior aligns with mean reversion theories, which posit that prices will revert to their historical averages after significant deviations."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_min, ts_max, ts_mean, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HighLowPriceRatioReversal(BaseFactor):
    factor_id = "high_low_price_ratio_reversal"
    name = "High-Low Price Ratio Reversal"
    category = "mean_reversion"
    description = "This factor computes the ratio of the current price to the high and low prices over a specified lookback period. The resulting ratio is then analyzed for extremes to signal potential reversals."
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)

        # Calculate the high-low price ratio
        high_low_ratio = close / (ts_max(high, 5) - ts_min(low, 5))
        high_low_ratio = high_low_ratio.replace([np.inf, -np.inf], np.nan)

        # Rank the high-low ratio to find extremes
        ranked_ratio = rank(high_low_ratio)

        return ranked_ratio