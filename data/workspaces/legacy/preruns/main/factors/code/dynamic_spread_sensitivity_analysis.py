"""The bid-ask spread is a key indicator of market liquidity, and its sensitivity to order flow can provide insights into market participants' intentions. By analyzing how the spread changes in response to variations in order flow, we can identify stocks that are likely to experience price movements based on shifts in market pressure."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, sign


@register_factor
class DynamicSpreadSensitivityAnalysis(BaseFactor):
    factor_id = "dynamic_spread_sensitivity_analysis"
    name = "Dynamic Spread Sensitivity Analysis"
    category = "microstructure"
    description = "This signal computes the relative change in the bid-ask spread in response to changes in order flow, effectively capturing the market's liquidity sensitivity."
    inputs = ["spread", "orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        spread = data["spread"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)

        # Calculate the rolling sum of order flow over the last 5 bars
        order_flow_sum = ts_sum(order_flow, 5)

        # Calculate the change in spread relative to the sum of order flow
        spread_change = delta(spread, 1)
        sensitivity = spread_change / (order_flow_sum.replace(0, np.nan))

        # Handle NaN values and ensure output has the same index and columns as spread
        return sensitivity.where(~sensitivity.isna(), 0.0)