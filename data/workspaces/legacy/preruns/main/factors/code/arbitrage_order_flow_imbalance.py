"""This factor exploits the notion that significant order flow imbalances indicate non-equilibrium conditions in the market, which may lead to short-term price corrections. By monitoring the disparity between buyer- and seller-initiated trades, we can identify potential arbitrage opportunities before the market self-corrects, especially after significant market events or announcements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, abs_, ts_mean


@register_factor
class ArbitrageOrderFlowImbalance(BaseFactor):
    factor_id = "arbitrage_order_flow_imbalance"
    name = "Arbitrage Order Flow Imbalance"
    category = "statistical_arbitrage"
    inputs = ["orderFlow", "volume"]
    description = "The signal computes the ratio of the absolute value of signed order flow (buy vs. sell) over a given time period, normalized by the total volume, across all stocks in the S&P 500. This provides a measure of imbalance that can indicate potential price movements in the near term."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate the absolute order flow
        abs_order_flow = abs_(order_flow)

        # Calculate the sum of order flow and volume over the last 5 bars
        sum_order_flow = ts_sum(abs_order_flow, 5)
        sum_volume = ts_sum(volume, 5)

        # Calculate the imbalance ratio
        imbalance_ratio = sum_order_flow / sum_volume

        # Handle NaN values by filling with 0.0
        return imbalance_ratio.fillna(0.0)