"""This factor leverages the sensitivity of limit order flow to price movements, particularly in volatile conditions. When limit orders are adjusted in response to price changes, it can indicate impending shifts in supply and demand dynamics, providing insight into potential price reversals or continuations. The concept aligns with the findings in the dynamic grid trading strategy paper, which emphasizes the importance of order flow in predicting market movements.

The signal computes the correlation between changes in limit order flow and subsequent price movements over a defined time frame, normalized by recent volatility. This captures how responsive the market is to incoming order flow, revealing potential trading opportunities.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import correlation, delta, ts_sum, stddev

@register_factor
class DynamicLimitOrderFlowSensitivity(BaseFactor):
    factor_id = "dynamic_limit_order_flow_sensitivity"
    name = "Dynamic Limit Order Flow Sensitivity"
    category = "microstructure"
    description = "Computes the correlation between changes in limit order flow and subsequent price movements, normalized by recent volatility."
    inputs = ["orderFlow", "close", "effSpread"]
    window_length = 60

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]
        eff_spread = data["effSpread"].fillna(0.0)

        # Calculate changes in order flow and price
        delta_order_flow = delta(order_flow, 1)
        delta_price = delta(close, 1)

        # Calculate rolling correlation
        correlation_signal = correlation(delta_order_flow, delta_price, 60)

        # Normalize by recent volatility (using effective spread as a proxy)
        volatility = stddev(eff_spread, 60)
        normalized_signal = correlation_signal / (volatility.replace(0, np.nan))

        return normalized_signal