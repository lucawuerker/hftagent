"""This factor is based on the idea that stocks exhibiting a high range of price movement during the day relative to their closing price are likely to continue trending in that direction. It builds upon the concept of momentum, suggesting that stocks showing significant intra-day volatility may attract momentum traders, leading to a self-reinforcing price movement. The source paper demonstrates that price range ratios can indicate potential for further price adjustment."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, ts_max, ts_min

@register_factor
class IntraDayPriceRangeRatio(BaseFactor):
    factor_id = "intra_day_price_range_ratio"
    name = "Intra-Day Price Range Ratio"
    category = "momentum"
    description = ("This signal computes the ratio of the intra-day price range (high - low) to the closing price for each stock, normalized over a specified period to identify potential momentum opportunities.")
    window_length = 60
    inputs = ["high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        close = data["close"].fillna(0.0)

        intra_day_range = high - low
        price_range_ratio = intra_day_range / close

        # Normalize over the specified window length
        normalized_ratio = ts_mean(price_range_ratio, self.window_length)

        return normalized_ratio