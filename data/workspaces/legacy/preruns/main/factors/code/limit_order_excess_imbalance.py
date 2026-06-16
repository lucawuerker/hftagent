"""Market participants tend to react to price movements by adjusting their limit orders, which can create imbalances in the order book. A significant excess in limit order imbalance, especially following price movements, can indicate potential price reversals as the imbalance corrects itself. This aligns with concepts from the microstructure literature, particularly in the context of self-exciting processes and market reactions to information."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean


@register_factor
class LimitOrderExcessImbalance(BaseFactor):
    factor_id = "limit_order_excess_imbalance"
    name = "Limit Order Excess Imbalance"
    category = "microstructure"
    inputs = ["orderFlow", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        # Calculate the excess imbalance
        excess_imbalance = order_flow / depth

        # Handle NaN values defensively
        excess_imbalance = excess_imbalance.fillna(0.0)

        # Calculate the rolling sum over the last 5 bars
        rolling_excess_imbalance = ts_sum(excess_imbalance, 5)

        return rolling_excess_imbalance