"""Research suggests that significant order flow imbalances can indicate impending price movements, as they reflect the relative strength of buyers versus sellers in the market. This factor leverages the concept of order flow imbalance, where a persistent positive or negative imbalance can signal forthcoming price trends, particularly in the context of short-term trading strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, sign

@register_factor
class HighOrderFlowImbalance(BaseFactor):
    factor_id = "high_order_flow_imbalance"
    name = "High Order Flow Imbalance"
    category = "microstructure"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate the rolling average of order flow over the last 5 bars
        rolling_order_flow = ts_sum(order_flow, 5)
        rolling_volume = ts_sum(volume, 5)

        # Normalize the rolling order flow by the rolling volume
        imbalance = rolling_order_flow / rolling_volume

        # Replace NaN values with 0.0 for the final output
        return imbalance.fillna(0.0)