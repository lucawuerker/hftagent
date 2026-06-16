"""Market participants tend to react differently to order flow depending on recent price movements and volatility. By analyzing how sensitive the price is to changes in order flow, we can gauge the market's current risk appetite and potential price impacts. This aligns with findings in microstructure literature that emphasize the importance of order flow in price formation."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (  
    delta, ts_sum, ts_mean, stddev, correlation,  
    abs_, sign, ts_rank,  
)

@register_factor
class DynamicOrderFlowSensitivityAnalysis(BaseFactor):
    factor_id = "dynamic_order_flow_sensitivity_analysis"
    name = "Dynamic Order Flow Sensitivity Analysis"
    category = "microstructure"
    description = "Computes the sensitivity of price changes to variations in order flow across a rolling window, weighted by recent volatility levels."
    inputs = ["close", "orderFlow", "effSpread", "effLobImb"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        order_flow = data["orderFlow"].fillna(0.0)
        eff_spread = data["effSpread"].replace(0, np.nan)
        eff_lob_imb = data["effLobImb"].replace(0, np.nan)

        # Calculate price changes and volatility
        price_change = delta(close, 1)
        volatility = stddev(close, 20)

        # Calculate rolling sums of order flow and price changes
        rolling_order_flow = ts_sum(order_flow, 20)
        rolling_price_change = ts_sum(price_change, 20)

        # Calculate sensitivity
        sensitivity = correlation(rolling_price_change, rolling_order_flow, 20)

        # Weight sensitivity by recent volatility
        weighted_sensitivity = sensitivity * volatility

        # Normalize the output
        output = ts_rank(weighted_sensitivity, 20)

        return output