"""This factor seeks to capture the resilience of the order flow in response to market shocks, which is indicative of market participants' willingness to provide liquidity. When order flow resilience is high, it implies a robust market that can absorb trades without significant price impact, suggesting a favorable environment for entry or exit points."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, ts_rank, abs_, delay


@register_factor
class DynamicOrderFlowResilience(BaseFactor):
    factor_id = "dynamic_order_flow_resilience"
    name = "Dynamic Order Flow Resilience"
    category = "microstructure"
    inputs = ["orderFlow", "effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        eff_spread = data["effSpread"].replace(0, np.nan)

        # Calculate the rolling sum of order flow over the last 5 bars
        rolling_order_flow_sum = ts_sum(order_flow, 5)
        # Calculate the rolling mean of effective spread over the last 5 bars
        rolling_eff_spread_mean = ts_mean(eff_spread, 5)

        # Calculate the resilience signal
        resilience_signal = rolling_order_flow_sum / rolling_eff_spread_mean

        # Normalize the resilience signal over a rolling window of 60 bars
        normalized_resilience = ts_rank(resilience_signal, 60)

        # Return the normalized resilience signal with the same index and columns as close
        return normalized_resilience