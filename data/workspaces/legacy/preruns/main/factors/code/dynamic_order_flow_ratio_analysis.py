"""This factor leverages the ratio of incoming order flow to overall trade volume to identify potential price movements. A high order flow ratio may indicate strong buying or selling pressure that precedes price changes, while a low ratio could suggest market indecision or lack of conviction among traders. This aligns with the notion that informed traders often utilize order flow to signal their intentions, prompting subsequent price adjustments."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, delay


@register_factor
class DynamicOrderFlowRatioAnalysis(BaseFactor):
    factor_id = "dynamic_order_flow_ratio_analysis"
    name = "Dynamic Order Flow Ratio Analysis"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate rolling sums
        order_flow_sum = ts_sum(order_flow, 5)
        volume_sum = ts_sum(volume, 5)

        # Calculate the dynamic order flow ratio
        order_flow_ratio = order_flow_sum / volume_sum

        # Handle NaN values in the ratio
        order_flow_ratio = order_flow_ratio.fillna(0.0)

        return order_flow_ratio