"""Building on the insights from the lead-lag effect observed in the NYSE, this factor utilizes order flow data to capture the influence of price movements in leading stocks on their lagging counterparts. Stocks that experience significant order flow pressure (either buying or selling) are expected to lead price changes in stocks that are correlated with them, thus creating actionable signals for trading strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (  
    correlation, ts_sum, ts_mean, abs_, sign, delay,  
)

@register_factor
class LeadLagOrderFlowSignal(BaseFactor):
    factor_id = "lead_lag_order_flow_signal"
    name = "Lead-Lag Order Flow Signal"
    category = "microstructure"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate the lagged order flow
        lagged_order_flow = delay(order_flow, 1)

        # Calculate the correlation of current order flow with lagged order flow
        correlation_signal = correlation(order_flow, lagged_order_flow, 60)

        # Handle NaN values by replacing them with 0
        correlation_signal = correlation_signal.fillna(0.0)

        return correlation_signal