"""This factor aims to capture the balance between aggressive buying and selling pressure, providing insight into market liquidity dynamics. When liquidity is constrained, aggressive buying can lead to price spikes, while excessive selling may drive prices down, indicating potential reversals or continuation patterns. The model is grounded in market microstructure theories that suggest liquidity imbalances directly affect price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ( 
    ts_sum, 
    ts_mean, 
    ts_rank, 
    abs_, 
    sign, 
    delay, 
)

@register_factor
class LiquidityPressureIndicator(BaseFactor):
    factor_id = "liquidity_pressure_indicator"
    name = "Liquidity Pressure Indicator"
    category = "microstructure"
    inputs = ["orderFlow", "effSpread", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        eff_spread = data["effSpread"].replace(0, np.nan)
        depth = data["depth"].fillna(0.0)

        # Calculate positive and negative order flow
        positive_flow = order_flow.where(order_flow > 0, 0)
        negative_flow = -order_flow.where(order_flow < 0, 0)

        # Calculate rolling sums
        sum_positive_flow = ts_sum(positive_flow, 5)
        sum_negative_flow = ts_sum(negative_flow, 5)

        # Calculate liquidity pressure ratio
        liquidity_pressure = sum_positive_flow / (sum_negative_flow + 1e-10)  # Avoid division by zero

        # Adjust by effective spread and depth
        adjusted_liquidity_pressure = liquidity_pressure * (depth / eff_spread.fillna(1.0))

        # Rank the result over a rolling window
        return ts_rank(adjusted_liquidity_pressure, 60).fillna(0.0)