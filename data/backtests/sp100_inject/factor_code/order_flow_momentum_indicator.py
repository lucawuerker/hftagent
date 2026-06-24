"""The Order Flow Momentum Indicator seeks to capitalize on the persistent directional influence of order flow imbalance on price movements. As highlighted in the paper on stochastic price dynamics, order flow imbalance can act as a market shock that leads to significant price drifts, particularly in the presence of momentum. By focusing on the cumulative order flow imbalance over a specified period, this factor aims to identify stocks with strong directional momentum following significant buying or selling pressure."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowMomentumIndicator(BaseFactor):
    factor_id = "order_flow_momentum_indicator"
    name = "Order Flow Momentum Indicator"
    category = "momentum"
    description = "This signal computes the cumulative order flow imbalance over a defined lookback period, normalized by the average volume during that period, to gauge the strength of recent buying or selling pressure relative to historical norms."
    window_length = 5
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        order_flow_imbalance = volume.where(volume > 0, -volume).rolling(window=self.window_length).sum()
        avg_volume = adv(volume, self.window_length)
        momentum_signal = order_flow_imbalance / avg_volume
        return momentum_signal.fillna(0.0)