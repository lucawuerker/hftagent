"""Volatility in order flow can signal potential price swings due to increased trading activity or uncertainty among market participants. A significant increase in the volatility of order flow may indicate that traders are reacting to new information, leading to potential price movements. This factor leans on the understanding that high volatility in order flow often precedes higher volatility in price, which can be exploited for timely trades."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import stddev, ts_mean


@register_factor
class OrderFlowVolatilityIndicator(BaseFactor):
    factor_id = "order_flow_volatility_indicator"
    name = "Order Flow Volatility Indicator"
    category = "volatility"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        order_flow_std = stddev(order_flow, 60)
        order_flow_mean = ts_mean(order_flow, 60)
        normalized_volatility = order_flow_std / order_flow_mean.replace(0, np.nan)
        return normalized_volatility