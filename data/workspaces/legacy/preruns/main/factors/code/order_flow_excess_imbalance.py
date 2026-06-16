"""This factor measures the deviation of order flow imbalance from its historical average. A significant excess imbalance may indicate that one side of the market (buy or sell) is aggressively pushing prices, thereby creating momentum which can lead to short-term price changes. This aligns with the self-exciting nature of order flows discussed in Hawkes process literature, where current order flows can predict future price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, delta, sign, adv


@register_factor
class OrderFlowExcessImbalance(BaseFactor):
    factor_id = "order_flow_excess_imbalance"
    name = "Order Flow Excess Imbalance"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate rolling average of order flow imbalance
        rolling_avg = ts_mean(order_flow, 60)

        # Calculate the current order flow imbalance deviation
        excess_imbalance = order_flow - rolling_avg

        # Normalize by average trading volume
        avg_volume = ts_mean(volume, 60)
        normalized_excess_imbalance = excess_imbalance / avg_volume.replace(0, np.nan)

        return normalized_excess_imbalance