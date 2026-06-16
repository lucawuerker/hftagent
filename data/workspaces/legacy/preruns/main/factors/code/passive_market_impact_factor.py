"""Based on the concept that passive limit orders can create price pressure in the market, this factor captures the imbalance between passive buy and sell orders at different price levels. As the volume of passive orders increases on one side, it is likely to push prices in the opposite direction, creating an opportunity for profit when the market adjusts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_rank, sign, abs_,)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PassiveMarketImpactFactor(BaseFactor):
    factor_id = "passive_market_impact_factor"
    name = "Passive Market Impact Factor"
    category = "microstructure"
    inputs = ["orderFlow", "depth"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        order_flow = data["orderFlow"].fillna(0.0)
        depth = data["depth"].replace(0, np.nan)

        # Calculate the ratio of passive buy and sell order flow
        passive_buy_flow = order_flow.where(order_flow > 0, 0)
        passive_sell_flow = -order_flow.where(order_flow < 0, 0)

        # Sum the passive flows over the last 5 bars
        sum_passive_buy = ts_sum(passive_buy_flow, 5)
        sum_passive_sell = ts_sum(passive_sell_flow, 5)

        # Calculate the market impact factor
        market_impact_factor = sum_passive_buy / (sum_passive_sell + 1e-10)  # Avoid division by zero

        # Normalize by depth to adjust for market conditions
        market_impact_factor = market_impact_factor / depth.fillna(1.0)

        return market_impact_factor.fillna(0.0)