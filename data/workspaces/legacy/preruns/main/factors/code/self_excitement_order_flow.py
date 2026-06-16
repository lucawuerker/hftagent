"""The self-exciting nature of order flow can indicate potential price movements, similar to models in behavioral finance that suggest that past trades influence future price actions. By analyzing the intensity and direction of order flow (buy vs. sell), we can capture how recent trades may trigger further trading activity, creating momentum in price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, sign


@register_factor
class SelfExcitementOrderFlow(BaseFactor):
    factor_id = "self_excitement_order_flow"
    category = "microstructure"
    inputs = ["orderFlow", "trade"]
    description = "This signal computes a cumulative measure of order flow, adjusting for the sign and volume of trades, to identify stocks with increasing self-excitement in trading activity. It analyzes the net order flow over the past few intervals to assess whether there is a persistent buying or selling pressure."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        trade = data["trade"].fillna(0.0)

        # Calculate net order flow over the last 5 intervals
        net_order_flow = ts_sum(order_flow, 5)

        # Calculate the sign of the trade volume
        sign_trade = sign(trade)

        # Combine the net order flow with the sign of the trades
        self_excitement = net_order_flow * sign_trade

        # Ensure no NaN values are present
        self_excitement = self_excitement.replace(0, np.nan)

        return self_excitement