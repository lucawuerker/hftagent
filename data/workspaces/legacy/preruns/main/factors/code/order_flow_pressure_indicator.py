"""Studies in market microstructure indicate that persistent order flow can create pressure on prices, leading to predictable price movements (Eisler et al., 2011). When there is a significant and consistent imbalance in order flow (either buy or sell), it often signals potential price trends or reversals due to the impact of liquidity on the price formation process."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean

@register_factor
class OrderFlowPressureIndicator(BaseFactor):
    factor_id = "order_flow_pressure_indicator"
    name = "Order Flow Pressure Indicator"
    category = "microstructure"
    description = "This factor computes the net signed order flow (orderFlow) over a defined lookback period, normalized by the total volume during that period. An increase in this indicator suggests growing pressure in the direction of the net order flow, while a decrease indicates diminishing pressure."
    inputs = ["orderFlow", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        volume = data["volume"].replace(0, np.nan)

        # Calculate net signed order flow over a 5-bar lookback period
        net_order_flow = ts_sum(order_flow, 5)
        total_volume = ts_sum(volume, 5)

        # Normalize the net order flow by total volume
        pressure_indicator = net_order_flow / total_volume

        # Handle potential NaN values resulting from division
        pressure_indicator = pressure_indicator.fillna(0.0)

        return pressure_indicator