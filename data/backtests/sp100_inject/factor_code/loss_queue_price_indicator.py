"""This factor leverages insights from queueing theory, suggesting that the number of losses during busy periods can indicate market stress. When there's a high number of losses, it may reflect increased demand and insufficient supply, leading to upward price pressure. This relationship is supported by research on stochastic inequalities in queueing systems, highlighting how losses correlate with price movements.

This signal computes the ratio of the number of losses during a busy period to the total number of trades, providing insights into market conditions. It is calculated using the traded volume and price data to capture the dynamics of price pressure linked to queueing systems."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean


@register_factor
class LossQueuePriceIndicator(BaseFactor):
    factor_id = "loss_queue_price_indicator"
    name = "Loss Queue Price Indicator"
    category = "other"
    description = "This signal computes the ratio of the number of losses during a busy period to the total number of trades, providing insights into market conditions."
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"].fillna(0.0)

        # Calculate the number of losses (where close is less than the previous close)
        losses = (close < close.shift(1)).astype(float)
        total_trades = volume

        # Calculate the sum of losses and total trades over a 5-bar window
        loss_sum = ts_sum(losses, 5)
        trade_sum = ts_sum(total_trades, 5)

        # Calculate the ratio of losses to total trades, avoiding division by zero
        ratio = loss_sum / trade_sum.replace(0, np.nan)

        return ratio