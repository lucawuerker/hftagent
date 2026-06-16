"""This factor leverages the relationship between limit order flow and market impact, which has been shown to exhibit concave behavior. By calculating the ratio of aggressive to passive limit orders during a given time period, we can gauge the potential market impact of trades and identify opportunities where the market may be underpricing or overpricing based on order flow dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delay

@register_factor
class LimitOrderImpactRatio(BaseFactor):
    factor_id = "limit_order_impact_ratio"
    name = "Limit Order Impact Ratio"
    category = "microstructure"
    description = "The signal computes the ratio of aggressive limit orders to the total limit orders (both aggressive and passive) over a rolling window. This ratio serves as a proxy for potential market impact and liquidity pressure."
    inputs = ["orderFlow", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        # Calculate aggressive and passive limit orders
        aggressive_orders = order_flow.where(order_flow > 0, 0.0)
        passive_orders = -order_flow.where(order_flow < 0, 0.0)

        # Total limit orders
        total_orders = aggressive_orders + passive_orders
        total_orders = total_orders.replace(0, np.nan)  # Avoid division by zero

        # Calculate the ratio of aggressive to total limit orders over a rolling window
        ratio = ts_sum(aggressive_orders, 5) / ts_sum(total_orders, 5)
        ratio = ratio.fillna(0.0)  # Fill NaN values with 0.0 for output

        return ratio