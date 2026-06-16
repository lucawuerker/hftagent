"""Market participants often overreact to recent order flow, leading to temporary price movements that can reverse. By analyzing sudden spikes in order flow and subsequent price action, we can identify potential mean-reversion opportunities, where prices are likely to revert back to their previous levels after an extreme order flow event. This behavior aligns with the principles of behavioral finance, which suggest that investors tend to overreact to news and trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, ts_rank, abs_, sign

@register_factor
class OrderFlowReversalSignal(BaseFactor):
    factor_id = "order_flow_reversal_signal"
    name = "Order Flow Reversal Signal"
    category = "mean_reversion"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate the sum of order flow over the last 5 bars
        order_flow_sum = ts_sum(order_flow, 5)

        # Calculate the price change over the last 5 bars
        price_change = delta(close, 5)

        # Identify reversal signals: when order flow is significantly positive and price change is negative
        reversal_signal = order_flow_sum.where(order_flow_sum > 0, 0) * sign(price_change)

        # Replace NaN with 0 for the final output
        return reversal_signal.fillna(0.0)