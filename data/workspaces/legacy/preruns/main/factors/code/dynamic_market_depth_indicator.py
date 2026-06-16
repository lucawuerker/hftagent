"""Market depth can provide insights into supply and demand dynamics, indicating potential price movements. By monitoring changes in the average top-of-book depth and its relationship with recent price movements, traders can gauge market sentiment and anticipate potential price reversals or continuations. This aligns with the notion that deeper markets may suggest stronger support or resistance levels."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, delta, ts_sum, sign


@register_factor
class DynamicMarketDepthIndicator(BaseFactor):
    factor_id = "dynamic_market_depth_indicator"
    name = "Dynamic Market Depth Indicator"
    category = "microstructure"
    description = "This factor computes the average top-of-book depth over a rolling window and compares it to the recent price changes to identify divergence or convergence signals."
    inputs = ["depth", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        depth = data["depth"].fillna(0.0)
        close = data["close"]

        # Calculate the average depth over the last 60 bars
        avg_depth = ts_mean(depth, 60)

        # Calculate the recent price changes
        price_change = delta(close, 1)

        # Calculate the signal: depth change relative to price change
        signal = avg_depth * sign(price_change)

        return signal.where(signal.notna(), 0.0)