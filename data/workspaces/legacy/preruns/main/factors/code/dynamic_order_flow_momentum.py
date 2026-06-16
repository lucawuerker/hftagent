"""Dynamic order flow can serve as a leading indicator of price movement, as increased buying or selling pressure often precedes significant price changes. By analyzing the momentum in order flow, traders can capture early signals of upcoming trends, particularly in volatile market conditions where order imbalances can drive price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_rank


@register_factor
class DynamicOrderFlowMomentum(BaseFactor):
    factor_id = "dynamic_order_flow_momentum"
    name = "Dynamic Order Flow Momentum"
    category = "momentum"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        momentum = delta(order_flow, 5)
        momentum_score = ts_rank(momentum, 60)
        return momentum_score