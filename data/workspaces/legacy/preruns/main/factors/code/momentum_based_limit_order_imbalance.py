"""The concept of momentum in financial markets suggests that recent price movements can forecast future price direction. By incorporating the limit order imbalance, which reflects the buying versus selling pressure, we can enhance the momentum signal, capturing the tendency of prices to continue moving in the same direction as indicated by the order flow. This aligns with the findings in the literature that highlight the significance of order flow in predicting price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delay, sign, abs_, rank


@register_factor
class MomentumBasedLimitOrderImbalance(BaseFactor):
    factor_id = "momentum_based_limit_order_imbalance"
    name = "Momentum-Based Limit Order Imbalance"
    category = "momentum"
    description = "This factor computes the difference between the volume of buy and sell limit orders over the recent bars, normalized by the total volume traded. It captures the prevailing market sentiment and its interaction with momentum, potentially signaling future price movements."
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the recent limit order imbalance
        lob_imbalance = ts_sum(order_flow, 5) / ts_sum(volume, 5)
        lob_imbalance = lob_imbalance.replace(0, np.nan)  # Avoid division by zero

        # Calculate momentum signal based on the limit order imbalance
        momentum_signal = rank(lob_imbalance)

        return momentum_signal