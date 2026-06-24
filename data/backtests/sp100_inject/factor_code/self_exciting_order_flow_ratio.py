"""Market dynamics exhibit self-exciting behavior, where past trades influence future order placements, leading to momentum or reversal in price movements. By measuring the ratio of buy to sell order flows over a recent period, we can capture the intensity of buying pressure relative to selling pressure, which can indicate potential price movements in the same direction. This factor leverages the insights from Hawkes processes, as outlined in the provided literature."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta

@register_factor
class SelfExcitingOrderFlowRatio(BaseFactor):
    factor_id = "self_exciting_order_flow_ratio"
    name = "Self-Exciting Order Flow Ratio"
    category = "microstructure"
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"].fillna(0.0)

        # Assuming volume represents the order flow where positive volume is buy and negative is sell
        buy_orders = volume.where(volume > 0, 0)
        sell_orders = -volume.where(volume < 0, 0)

        # Calculate the rolling sums of buy and sell orders
        buy_flow = ts_sum(buy_orders, 5)
        sell_flow = ts_sum(sell_orders, 5)

        # Calculate the total order flow
        total_flow = buy_flow + sell_flow
        total_flow = total_flow.replace(0, np.nan)  # Avoid division by zero

        # Calculate the ratio of buy to sell order flows
        order_flow_ratio = buy_flow / total_flow

        return order_flow_ratio