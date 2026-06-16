"""Order flow has been shown to lead price movements, as it reflects the immediate market sentiment and the aggressiveness of buyers and sellers. By analyzing the ratio of positive to negative order flow over recent bars, we can capture the momentum of underlying market pressure, which may predict future price movements. This idea aligns with the general principles outlined in momentum trading literature, particularly in the context of futures and equities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, sign, ts_mean

@register_factor
class MomentumOrderFlowRatio(BaseFactor):
    factor_id = "momentum_order_flow_ratio"
    name = "Order Flow Momentum Ratio"
    category = "momentum"
    description = "This factor computes the ratio of the cumulative positive order flow to cumulative negative order flow over a specified lookback period, normalized to account for volatility in the order flow signal."
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate cumulative positive and negative order flow
        positive_of = order_flow.where(order_flow > 0, 0).rolling(window=5).sum()
        negative_of = -order_flow.where(order_flow < 0, 0).rolling(window=5).sum()

        # Calculate the ratio of positive to negative order flow
        ratio = positive_of / (negative_of.replace(0, np.nan))  # Avoid division by zero

        # Normalize the ratio by the average volume to account for volatility
        normalized_ratio = ratio / ts_mean(volume, 5).replace(0, np.nan)

        return normalized_ratio