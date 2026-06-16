"""The presence of limit orders can indicate where traders are willing to buy or sell, influencing price movement. A strong imbalance between buy and sell limit orders, particularly when it changes dynamically, can signal impending price movements as liquidity shifts and traders react to the market's order book structure."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, sign


@register_factor
class DynamicLimitOrderImbalance(BaseFactor):
    factor_id = "dynamic_limit_order_imbalance"
    name = "Dynamic Limit Order Imbalance"
    category = "microstructure"
    inputs = ["orderFlow", "depth"]
    description = "This signal computes the difference between the cumulative limit order volume on the buy side and the sell side, normalized by the total limit order volume at the top of the book. It captures the shifts in market sentiment and order pressure, providing insights into potential price direction."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        # Calculate cumulative buy and sell order flow
        cumulative_buy = ts_sum(order_flow.where(order_flow > 0, 0), 5)
        cumulative_sell = ts_sum(order_flow.where(order_flow < 0, 0), 5)

        # Normalize by total depth
        total_depth = ts_sum(depth, 5).fillna(1)  # Avoid division by zero
        imbalance = (cumulative_buy + abs_(cumulative_sell))

        # Calculate dynamic limit order imbalance
        dynamic_imbalance = (cumulative_buy - abs_(cumulative_sell)) / total_depth

        return dynamic_imbalance