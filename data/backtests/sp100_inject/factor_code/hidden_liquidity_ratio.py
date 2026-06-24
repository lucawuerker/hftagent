"""The presence of hidden liquidity can significantly impact price movements, as traders often do not see the full depth of orders in the market. By analyzing the ratio of executed trades to the resting limit orders at a given price level, we can infer the level of hidden liquidity, which may indicate potential price reversals or continuations as liquidity is revealed. This is particularly relevant in fast-moving markets where hidden orders can alter the supply-demand balance unexpectedly."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, adv


@register_factor
class HiddenLiquidityRatio(BaseFactor):
    factor_id = "hidden_liquidity_ratio"
    name = "Hidden Liquidity Ratio"
    category = "microstructure"
    description = "This signal computes the ratio of total executed volume to the total volume of resting limit orders at each price level over a specified time period, providing insight into the presence of hidden liquidity in the market."
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]
        adv5 = adv(volume, 5)
        total_executed_volume = ts_sum(volume, 5)
        hidden_liquidity_ratio = total_executed_volume / adv5.replace(0, np.nan)
        return hidden_liquidity_ratio.where(adv5 > 0, 0.0)