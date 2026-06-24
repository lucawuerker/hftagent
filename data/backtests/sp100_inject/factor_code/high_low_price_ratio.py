"""Stocks that exhibit a strong upward momentum often show large differences between their high and low prices within a given period, reflecting increasing volatility and investor interest. This price action can indicate the potential for continued upward movement as traders chase rising stocks, capturing the momentum effect. The idea aligns with behavioral finance principles where traders react strongly to recent price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_max, ts_min, sign, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HighLowPriceRatio(BaseFactor):
    factor_id = "high_low_price_ratio"
    name = "High-Low Price Ratio"
    category = "momentum"
    inputs = ["high", "low"]
    description = "This factor computes the ratio of the intra-bar high price to the intra-bar low price for each stock, capturing the relative strength of price movements over a specific time period."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].replace(0, np.nan)
        ratio = high / low
        return ratio.where(low.notna(), 0.0).rank(axis=1)