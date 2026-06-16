"""This factor captures the relationship between order flow and liquidity conditions in the market. When order flow is strong and liquidity is high, it indicates confidence from market participants, leading to potential upward price momentum. Conversely, weak order flow amidst low liquidity can signal bearish conditions, suggesting potential price declines. The signal computes a ratio of the signed order flow to the liquidity measure (depth or order-flow liquidity proxy). This ratio is then smoothed over a rolling window to identify shifts in market sentiment and liquidity conditions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, sign, delay

@register_factor
class OrderFlowLiquiditySignal(BaseFactor):
    factor_id = "order_flow_liquidity_signal"
    name = "Order Flow Liquidity Signal"
    category = "microstructure"
    inputs = ["orderFlow", "ofLiq", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        of_liq = data["ofLiq"].replace(0, np.nan)
        depth = data["depth"].fillna(0.0)

        # Calculate the ratio of signed order flow to order-flow liquidity proxy
        ratio = order_flow / of_liq
        ratio = ratio.where(of_liq.notna(), 0.0)  # Replace NaN with 0 where of_liq is NaN

        # Smooth the ratio over a rolling window of 60 bars
        smoothed_ratio = ts_mean(ratio, 60)

        return smoothed_ratio