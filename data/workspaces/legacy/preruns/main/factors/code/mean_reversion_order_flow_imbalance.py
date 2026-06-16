"""Order flow imbalance can signal short-term price movements. When the order flow becomes excessively skewed in one direction, it often indicates an overreaction by market participants, leading to a potential mean-reversion opportunity. This aligns with behavioral finance theories suggesting that traders tend to overreact to news, creating opportunities for contrarian strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, sign, ts_mean, abs_, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MeanReversionOrderFlowImbalance(BaseFactor):
    factor_id = "mean_reversion_order_flow_imbalance"
    name = "Mean Reversion Order Flow Imbalance"
    category = "mean_reversion"
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the rolling sum of order flow over the last 5 bars
        order_flow_sum = ts_sum(order_flow, 5)
        # Calculate the rolling mean of order flow over the last 20 bars
        order_flow_mean = ts_mean(order_flow, 20)

        # Calculate the deviation from the mean
        deviation = order_flow_sum - order_flow_mean

        # Normalize the deviation by the average volume to get a signal
        normalized_signal = deviation / (volume.replace(0, np.nan).fillna(1.0))

        # Return the normalized signal, ensuring the same index and columns as close
        return normalized_signal.where(~normalized_signal.isna(), 0.0).fillna(0.0)