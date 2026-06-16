"""This factor identifies stocks that exhibit significant price movements in response to changes in order flow dynamics. As evidenced in microstructure literature, stocks with high sensitivity to order flow are likely to experience larger price adjustments, particularly in volatile environments, thus creating potential trading opportunities.

The factor computes the sensitivity of price changes to fluctuations in order flow, using the orderFlow and price data from the LOBSTER-derived bars. A higher sensitivity indicates a stronger relationship between order flow changes and subsequent price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, ts_mean, correlation, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowSensitivityAnalysis(BaseFactor):
    factor_id = "order_flow_sensitivity_analysis"
    name = "Order Flow Sensitivity Analysis"
    category = "microstructure"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate price changes
        price_change = delta(close, 1).fillna(0.0)

        # Calculate rolling mean and stddev of order flow
        mean_order_flow = ts_mean(order_flow, 10).fillna(0.0)
        std_order_flow = stddev(order_flow, 10).fillna(0.0)

        # Calculate sensitivity as correlation between price changes and order flow
        sensitivity = correlation(price_change, order_flow, 10).fillna(0.0)

        # Normalize sensitivity by the standard deviation of order flow to get a relative measure
        normalized_sensitivity = sensitivity / (std_order_flow.replace(0, np.nan))

        # Replace NaN with 0 for final output
        return normalized_sensitivity.fillna(0.0)