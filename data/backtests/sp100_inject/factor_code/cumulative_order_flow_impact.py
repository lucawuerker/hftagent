"""The price impact of trades is influenced by the net order flow imbalance, which aggregates the effect of both market and limit orders on price changes over short time intervals. By measuring the cumulative order flow impact, we can capture the sustained effect of recent trading activity on current prices, reflecting the supply-demand dynamics that drive price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CumulativeOrderFlowImpact(BaseFactor):
    factor_id = "cumulative_order_flow_impact"
    name = "Cumulative Order Flow Impact"
    category = "microstructure"
    description = "This factor computes the cumulative order flow impact as the rolling sum of the order flow imbalance over a specified lookback period, capturing how past trading activity influences current price behavior."
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        order_flow_imbalance = delta(close, 1) * volume
        cumulative_impact = ts_sum(order_flow_imbalance, 5)
        return cumulative_impact