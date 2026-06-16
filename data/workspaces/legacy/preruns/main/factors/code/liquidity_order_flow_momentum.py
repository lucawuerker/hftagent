"""Liquidity dynamics play a crucial role in price formation, as higher liquidity often indicates stronger market confidence. By analyzing the order flow alongside liquidity metrics, this factor aims to capture momentum driven by increasing liquidity, which can lead to more significant price movements in the same direction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, correlation, ts_mean, abs_, sign)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquidityOrderFlowMomentum(BaseFactor):
    factor_id = "liquidity_order_flow_momentum"
    name = "Liquidity Order Flow Momentum"
    category = "momentum"
    inputs = ["orderFlow", "depth", "effSpread", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].fillna(0.0)
        eff_spread = data["effSpread"].replace(0, np.nan)
        close = data["close"]

        # Calculate the correlation between order flow and subsequent price movements
        future_close = delta(close, 1).fillna(0.0)
        liquidity_signal = correlation(order_flow, future_close, 10)

        # Calculate the average depth and effective spread
        avg_depth = ts_mean(depth, 10)
        avg_eff_spread = ts_mean(eff_spread, 10)

        # Combine signals: higher liquidity with positive order flow indicates momentum
        momentum_signal = liquidity_signal.where(avg_depth > 0, 0.0) * sign(order_flow)

        return momentum_signal