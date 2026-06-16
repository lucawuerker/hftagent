"""Order flow imbalances often lead to momentum effects where price trends can persist in the direction of the flow. However, when extreme order flow occurs, it can signify an overextension, leading to a reversal in momentum as the market corrects. This aligns with behavioral finance theories that suggest traders often overreact to new information, creating temporary mispricings.

This factor computes the momentum reversal signal by analyzing the net signed order flow over a defined period, identifying extreme values that indicate potential price reversals. The signal is constructed by assessing the correlation between the order flow and subsequent price movements, capturing the potential for counter-movements following substantial order flow events.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (  
    delta, ts_sum, correlation, ts_rank, abs_, sign,  
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OrderFlowMomentumReversal(BaseFactor):
    factor_id = "order_flow_momentum_reversal"
    name = "Order Flow Momentum Reversal"
    category = "momentum"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]

        # Calculate the 5-bar sum of order flow
        order_flow_sum = ts_sum(order_flow, 5)

        # Calculate the 5-bar delta of close prices
        close_delta = delta(close, 5)

        # Calculate the correlation between order flow and close delta over the last 20 bars
        correlation_signal = correlation(order_flow_sum, close_delta, 20)

        # Rank the correlation signal to normalize it
        ranked_signal = ts_rank(correlation_signal, 60)

        # Return the final signal, ensuring no NaN values
        return ranked_signal.where(ranked_signal.notna(), 0.0)