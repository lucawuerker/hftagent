"""Limit order pressure can signal impending price movements. When there is a significant increase in limit orders on one side of the order book, it suggests that traders expect price changes, potentially leading to price reversals. This factor aims to capture such dynamics by analyzing the net limit order flow in real-time."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_rank, abs_, sign)

@register_factor
class HighFrequencyLimitOrderPressure(BaseFactor):
    factor_id = "high_frequency_limit_order_pressure"
    name = "High Frequency Limit Order Pressure"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate rolling sum of order flow and volume
        rolling_order_flow_sum = ts_sum(order_flow, 5)
        rolling_volume_sum = ts_sum(volume, 5)

        # Normalize order flow by volume
        normalized_order_flow = rolling_order_flow_sum / rolling_volume_sum.replace(0, np.nan)

        # Rank the normalized order flow
        return ts_rank(normalized_order_flow, 60)