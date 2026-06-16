"""Market participants often place limit orders that can indicate their expectations for future price movements. A discrepancy between the volume of limit orders added on the bid side versus the ask side can signal bullish or bearish sentiment, impacting future price changes. This factor aims to capture those moments when there is an imbalance in limit order pressure, suggesting potential price direction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, sign


@register_factor
class LimitOrderPressureRatio(BaseFactor):
    factor_id = "limit_order_pressure_ratio"
    name = "Limit Order Pressure Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of limit order flow into the bid side versus the ask side, normalized by the total limit order flow. A higher ratio indicates stronger buying pressure, while a lower ratio suggests selling pressure."
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        bid_flow = order_flow.where(order_flow > 0, 0.0)
        ask_flow = -order_flow.where(order_flow < 0, 0.0)
        total_flow = ts_sum(order_flow, 5).replace(0, np.nan)

        ratio = bid_flow / total_flow
        return ratio.fillna(0.0)