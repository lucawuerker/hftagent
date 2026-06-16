"""Order flow efficiency captures how effectively market participants are utilizing their trading strategies in relation to market liquidity. By analyzing the relationship between order flow and price movements, this factor can identify stocks where traders are able to exert significant influence on price direction, potentially leading to profitable trading opportunities. This idea is inspired by concepts of liquidity and market impact discussed in the literature on order flow dynamics."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import delta, ts_sum

@register_factor
class DynamicOrderFlowEfficiency(BaseFactor):
    factor_id = "dynamic_order_flow_efficiency"
    name = "Dynamic Order Flow Efficiency"
    category = "microstructure"
    inputs = ["orderFlow", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        close = data["close"]
        price_change = delta(close, 1).replace(0, np.nan)
        efficiency = order_flow / price_change
        return efficiency.fillna(0.0)