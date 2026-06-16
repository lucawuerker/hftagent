"""Order flow often indicates the direction of price movement due to its impact on supply and demand dynamics. When there's a persistent positive (or negative) order flow, it reflects strong buyer (or seller) interest, suggesting potential upward (or downward) price momentum. This aligns with findings that highlight the predictive power of order flow on future returns. This factor computes the cumulative order flow over a defined period (e.g., the last 30 minutes) for each stock. A higher cumulative order flow is expected to correlate with future positive returns, while negative cumulative order flow may predict negative returns."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delay
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowMomentum(BaseFactor):
    factor_id = "order_flow_momentum"
    name = "Order Flow Momentum"
    category = "momentum"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        cumulative_order_flow = ts_sum(order_flow, 18)  # 30 minutes = 3 bars of 10 seconds each
        return cumulative_order_flow.where(cumulative_order_flow != 0, np.nan)