"""Limit order flow can indicate potential price direction due to the pressure it exerts on the market. When there is a consistent increase in net limit orders (either on the buy or sell side), it suggests a growing trend that could lead to price momentum, particularly in the direction of the prevailing order flow. This signal computes the momentum of net limit order flow over a defined window, capturing trends in the buying or selling pressure. It is derived from the 'orderFlow' field, focusing on periods where limit orders are predominantly added or removed."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, sign


@register_factor
class LimitOrderFlowMomentum(BaseFactor):
    factor_id = "limit_order_flow_momentum"
    name = "Limit Order Flow Momentum"
    category = "momentum"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        momentum_window = 60
        momentum = ts_sum(order_flow, momentum_window)
        momentum = momentum.replace(0, np.nan)
        return momentum.where(momentum > 0, 0.0)