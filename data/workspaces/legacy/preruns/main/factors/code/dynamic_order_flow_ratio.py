"""The Dynamic Order Flow Ratio captures the relationship between aggressive buy and sell orders, providing insights into market sentiment and potential price direction. When the ratio of buy to sell order flow increases significantly, it may indicate bullish sentiment, while a decline may suggest bearish sentiment. This aligns with behavioral finance theories that emphasize how order flow reflects trader intentions and market psychology."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, abs_, sign, ts_mean


@register_factor
class DynamicOrderFlowRatio(BaseFactor):
    factor_id = "dynamic_order_flow_ratio"
    name = "Dynamic Order Flow Ratio"
    category = "microstructure"
    inputs = ["orderFlow"]
    description = "This signal computes the ratio of the absolute value of buy order flow to the absolute value of sell order flow over a specified rolling window, normalized to a standard deviation to detect significant deviations from the mean. A higher ratio indicates a stronger buying pressure relative to selling pressure."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        buy_order_flow = order_flow.where(order_flow > 0, 0.0)
        sell_order_flow = -order_flow.where(order_flow < 0, 0.0)

        sum_buy = ts_sum(buy_order_flow, 5)
        sum_sell = ts_sum(sell_order_flow, 5)

        ratio = sum_buy / (sum_sell.replace(0, np.nan))
        ratio = ratio.fillna(0.0)

        mean_ratio = ts_mean(ratio, 60)
        std_ratio = ts_mean((ratio - mean_ratio) ** 2, 60) ** 0.5

        normalized_ratio = (ratio - mean_ratio) / std_ratio
        return normalized_ratio