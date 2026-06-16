"""High volatility in order flow can indicate market uncertainty and potential price movements. This factor leverages the premise that increased order flow volatility can lead to larger price swings, as traders react to rapidly changing conditions. By capturing these bursts of order flow activity, traders can position themselves ahead of potential market movements.

This signal computes the rolling standard deviation of the order flow over a specified window, normalized by the average order flow. It highlights periods of heightened activity and potential market impact, providing a measure of liquidity and sentiment.

"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, stddev, abs_


@register_factor
class OrderFlowVolatility(BaseFactor):
    factor_id = "order_flow_volatility"
    name = "Order Flow Volatility Indicator"
    category = "microstructure"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        mean_order_flow = ts_mean(order_flow, 20)
        std_order_flow = stddev(order_flow, 20)
        normalized_volatility = std_order_flow / mean_order_flow.replace(0, np.nan)
        return normalized_volatility