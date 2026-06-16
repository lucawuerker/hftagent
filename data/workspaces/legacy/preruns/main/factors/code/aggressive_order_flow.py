"""Aggressive orders, which are typically market orders, indicate a strong conviction from traders about the direction of price. By analyzing the ratio of aggressive to passive orders, we can gauge market sentiment and potential price movements. This factor leverages the imbalance in order flow to predict short-term price changes, as indicated in studies on order book dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, sign


@register_factor
class AggressiveOrderFlow(BaseFactor):
    factor_id = "aggressive_order_flow"
    name = "Aggressive Order Flow Factor"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate the total order flow over the last 5 bars
        total_order_flow = ts_sum(order_flow, 5)
        # Calculate the aggressive order flow (net market orders)
        aggressive_order_flow = delta(order_flow, 1)

        # Calculate the ratio of aggressive to total order flow
        ratio = aggressive_order_flow / total_order_flow
        ratio = ratio.fillna(0.0)  # Fill NaN values with 0.0

        return ratio