"""Liquidity fluctuations can signal impending price movements. When liquidity suddenly decreases, it may indicate that market participants are pulling back, potentially leading to price declines, while increased liquidity can signal confidence in price stability. This aligns with findings from limit order book models that emphasize the relationship between liquidity and price dynamics.

This factor computes the ratio of the change in average top-of-book depth to the change in average volume traded over a given period, normalized by the average liquidity over the same period. It captures the relative strength of liquidity fluctuations in the market.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, ts_mean, ts_sum, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquidityFluctuationIndicator(BaseFactor):
    factor_id = "liquidity_fluctuation_indicator"
    name = "Liquidity Fluctuation Indicator"
    category = "microstructure"
    inputs = ["depth", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        depth = data["depth"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        avg_depth = ts_mean(depth, 5)
        avg_volume = ts_mean(volume, 5)

        change_depth = delta(avg_depth, 1)
        change_volume = delta(avg_volume, 1)

        normalized_liquidity = avg_depth / (avg_volume.replace(0, np.nan))
        liquidity_fluctuation = change_depth / change_volume

        return liquidity_fluctuation.where(~normalized_liquidity.isna(), 0.0)