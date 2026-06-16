"""This factor captures the relationship between order flow and bid-ask spread dynamics. Given that increased order flow can lead to tighter spreads, understanding how spread changes in response to order flow can signal potential price movements. Market participants often react to changes in liquidity and order flow, making this a potentially predictive signal for short-term price changes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import delta, ts_sum

@register_factor
class OrderFlowSpreadSensitivity(BaseFactor):
    factor_id = "order_flow_spread_sensitivity"
    name = "Order Flow Spread Sensitivity"
    category = "microstructure"
    inputs = ["spread", "orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        spread = data["spread"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)

        spread_change = delta(spread, 1)
        order_flow_change = delta(order_flow, 1)

        # Replace zeros in order_flow_change to avoid division by zero
        order_flow_change = order_flow_change.replace(0, np.nan)

        # Calculate sensitivity
        sensitivity = spread_change / order_flow_change

        # Return the result with the same index and columns as the input
        return sensitivity