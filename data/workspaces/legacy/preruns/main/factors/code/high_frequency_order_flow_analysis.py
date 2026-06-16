"""High-frequency trading strategies often leverage order flow dynamics to predict short-term price movements. By analyzing the net order flow (the difference between buy and sell orders) at a granular level, traders can identify imminent price shifts caused by aggressive buying or selling pressures, which can provide an edge in execution timing. This approach aligns with findings in market microstructure literature suggesting that order flow is a leading indicator of price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, sign

@register_factor
class HighFrequencyOrderFlowAnalysis(BaseFactor):
    factor_id = "high_frequency_order_flow_analysis"
    name = "High-Frequency Order Flow Analysis"
    category = "microstructure"
    description = "This factor computes the net order flow for each stock by aggregating the signed traded volume over a specified recent window."
    inputs = ["trade"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        net_order_flow = ts_sum(trade, 5)  # Aggregating over the last 5 bars
        return net_order_flow.where(net_order_flow != 0, np.nan)  # Replace zeros with NaN