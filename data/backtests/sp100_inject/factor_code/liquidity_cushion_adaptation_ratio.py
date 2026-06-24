"""The liquidity cushion around the midpoint price adjusts rapidly to price movements, indicating the responsiveness of market participants to shifts in supply and demand. As noted in the referenced study, a well-defined and adaptive liquidity cushion can mitigate the impact of price shocks and enhance market stability. Stocks with a more dynamic liquidity cushion may outperform in volatile conditions as they can better absorb large trades without significant price swings."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LiquidityCushionAdaptationRatio(BaseFactor):
    factor_id = "liquidity_cushion_adaptation_ratio"
    name = "Liquidity Cushion Adaptation Ratio"
    category = "microstructure"
    description = ("This signal calculates the ratio of the volume of limit orders within a defined liquidity cushion to the total volume of limit orders at various price levels over time. "
                   "It provides insights into how well the liquidity cushion adapts to price changes, reflecting market participants' readiness to react.")
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]

        # Define a liquidity cushion as a percentage of the close price
        liquidity_cushion = close * 0.01  # Example: 1% of the close price

        # Calculate the volume within the liquidity cushion
        volume_within_cushion = volume.where(volume <= liquidity_cushion, 0)

        # Calculate the total volume
        total_volume = ts_sum(volume, 5)

        # Calculate the ratio
        ratio = ts_sum(volume_within_cushion, 5) / total_volume.replace(0, np.nan)

        return ratio