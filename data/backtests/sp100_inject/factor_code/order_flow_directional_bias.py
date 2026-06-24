"""This factor exploits the tendency of price movements to follow order flow imbalances, which can indicate the sentiment of market participants. The source paper suggests that when there is a persistent directional bias in order flow, it often leads to subsequent price trends, allowing traders to capitalize on these short-term movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, sign


@register_factor
class OrderFlowDirectionalBias(BaseFactor):
    factor_id = "order_flow_directional_bias"
    name = "Order Flow Directional Bias"
    category = "microstructure"
    description = ("The signal computes the ratio of net buy to net sell orders over a defined period, capturing the directional bias in order flow. "
                   "A higher ratio indicates bullish sentiment, while a lower ratio signals bearish sentiment, providing a predictive signal for price movements.")
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        net_buy = volume.where(volume > 0, 0).rolling(window=5).sum()
        net_sell = volume.where(volume < 0, 0).rolling(window=5).sum()
        directional_bias = net_buy / (net_sell.replace(0, np.nan))  # Avoid division by zero
        return directional_bias.fillna(0.0)