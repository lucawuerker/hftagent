"""Order flow reflects the immediate buying and selling pressure in the market, and understanding its momentum helps capture price trends before they fully materialize. As shown in Choi's paper, price momentum can be amplified by the volume of trades, suggesting that stocks with increasing buy order flow tend to exhibit continued upward price momentum."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, delta, sign

@register_factor
class MomentumBasedOrderFlow(BaseFactor):
    factor_id = "momentum_based_order_flow"
    name = "Momentum Based Order Flow"
    category = "momentum"
    description = "This factor computes the momentum of the order flow by taking the difference of the current order flow and its moving average over a specified window. A positive value indicates growing buying pressure, while a negative value indicates selling pressure."
    inputs = ["orderFlow"]
    window_length = 20

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        order_flow_ma = ts_mean(order_flow, 20)
        momentum = delta(order_flow, 1) - order_flow_ma
        return momentum