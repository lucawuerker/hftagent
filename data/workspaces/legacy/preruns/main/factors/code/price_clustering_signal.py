"""The price clustering phenomenon indicates that traders tend to favor certain price levels, especially those that are round numbers or multiples of specific tick sizes. This behavior is often driven by market psychology, where traders exhibit a preference for placing orders at these levels, leading to increased trading activity and price stability around these points. Understanding and leveraging this behavior can provide an edge in identifying potential price reversals or breakout points."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_argmax, ts_rank


@register_factor
class PriceClusteringSignal(BaseFactor):
    factor_id = "price_clustering_signal"
    name = "Price Clustering Signal"
    category = "microstructure"
    description = "Computes the frequency of trades at rounded prices normalized by total trades."
    inputs = ["trade", "nbTrades"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        nb_trades = data["nbTrades"].fillna(1.0)  # Avoid division by zero

        # Define rounded price levels (e.g., multiples of 0.05)
        rounded_prices = (trade.index.astype(int) // 5) * 5

        # Count trades at rounded prices
        trade_counts = trade.groupby(rounded_prices).sum().reindex(trade.index, fill_value=0)

        # Normalize by total trades
        frequency = trade_counts / nb_trades

        # Fill NaN values with 0
        frequency = frequency.fillna(0.0)

        return frequency