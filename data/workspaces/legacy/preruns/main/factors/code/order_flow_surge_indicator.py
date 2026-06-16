"""Sudden spikes in order flow can indicate a shift in market sentiment and potential price movements. The presence of aggressive buying or selling pressure often precedes significant price changes, as it reflects the intentions of informed traders acting on new information. This factor leverages the principle that large, unexpected order flows can serve as a leading indicator of future price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowSurgeIndicator(BaseFactor):
    factor_id = "order_flow_surge_indicator"
    name = "Order Flow Surge Indicator"
    category = "microstructure"
    description = "This signal computes the ratio of the current bar's signed traded volume to a rolling average of the previous bars' signed traded volumes. A high ratio indicates a surge in order flow, suggesting potential price momentum."
    inputs = ["trade"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        rolling_avg_trade = ts_sum(trade, 5).replace(0, np.nan)
        current_trade = trade
        order_flow_surge = current_trade / rolling_avg_trade
        return order_flow_surge