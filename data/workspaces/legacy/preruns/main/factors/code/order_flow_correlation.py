"""High-frequency trading strategies often exploit the correlation between order flow dynamics of related assets. When two assets exhibit similar order flow patterns, it may indicate a shared underlying driver, leading to potential price movements in the same direction. This factor aims to capture such correlations to identify trading opportunities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowCorrelation(BaseFactor):
    factor_id = "order_flow_correlation"
    name = "Order Flow Correlation"
    category = "microstructure"
    inputs = ["orderFlow"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        correlation_signal = correlation(order_flow, order_flow, 60)
        return correlation_signal