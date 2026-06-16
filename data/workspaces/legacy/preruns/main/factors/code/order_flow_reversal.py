"""In a market where order flow is predominantly driven by aggressive buyers or sellers, an abrupt reversal in order flow can indicate a potential price reversal. This is based on the premise that significant shifts in order flow often precede changes in market sentiment and price direction, as suggested by microstructure literature on liquidity and order dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, sign, ts_rank


@register_factor
class OrderFlowReversal(BaseFactor):
    factor_id = "order_flow_reversal"
    name = "Order Flow Reversal Indicator"
    category = "microstructure"
    description = "This factor computes the rate of change in order flow, measured by the signed traded volume over a rolling window, and identifies instances when the order flow switches from positive to negative or vice versa."
    inputs = ["trade"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        order_flow_change = delta(trade, 1)
        rolling_sum = ts_sum(order_flow_change, 5)
        standardized_signal = ts_rank(rolling_sum, 60)
        return standardized_signal.where(rolling_sum != 0, np.nan)