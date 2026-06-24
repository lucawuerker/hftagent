"""Order flow reflects the balance of buyers and sellers in the market and can provide insights into future price movements. As highlighted in the literature, momentum strategies that leverage order flow information can lead to superior returns, especially in high-frequency trading contexts where immediate market reactions matter."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowBasedMomentum(BaseFactor):
    factor_id = "order_flow_based_momentum"
    name = "Order Flow Based Momentum"
    category = "momentum"
    description = ("This factor computes the cumulative signed order flow (buy orders minus sell orders) over a specified lookback period, "
                   "normalized by the average volume traded during that period. A positive cumulative order flow indicates upward momentum, "
                   "while a negative value suggests downward momentum.")
    window_length = 60
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        signed_order_flow = volume * (close.diff().fillna(0.0).where(close.diff() > 0, 0) - close.diff().fillna(0.0).where(close.diff() < 0, 0))
        cumulative_order_flow = ts_sum(signed_order_flow, self.window_length)
        normalized_order_flow = cumulative_order_flow / adv(volume, self.window_length)
        return normalized_order_flow