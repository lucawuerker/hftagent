"""The limit order flow ratio, defined as the ratio of signed limit order flow to total traded volume, can indicate the market's inclination towards buying or selling. An increased ratio suggests aggressive buying pressure, while a decrease may indicate selling pressure, making it a useful signal to capture short-term price movements before they materialize. This aligns with the notion that traders often react to limit order flow as a lead indicator of price trends.

This factor computes the ratio of net limit order flow to the total traded volume over a specified period, providing insights into market sentiment and potential price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean

@register_factor
class LimitOrderFlowRatioAnalysis(BaseFactor):
    factor_id = "limit_order_flow_ratio_analysis"
    name = "Limit Order Flow Ratio Analysis"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate the total traded volume over the last 5 bars
        total_volume = ts_sum(volume, 5)

        # Calculate the net limit order flow over the last 5 bars
        net_order_flow = ts_sum(order_flow, 5)

        # Calculate the limit order flow ratio
        ratio = net_order_flow / total_volume

        # Handle NaN values in the ratio
        ratio = ratio.fillna(0.0)

        return ratio