"""Trades executed by traders located in the core of the trading network (k-shell) tend to have a higher immediate price impact compared to those from peripheral traders. This aligns with findings that institutional traders, who are often in the core, execute larger orders that significantly influence market prices. Understanding the impact of trader positions can provide insights into future price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import (  
    delta, ts_sum, ts_mean, ts_rank, sign, abs_,  
)

@register_factor
class TraderPositionImpactAnalysis(BaseFactor):
    factor_id = "trader_position_impact_analysis"
    name = "Trader Position Impact Analysis"
    category = "microstructure"
    inputs = ["trade", "close", "volume", "effSpread", "lobImb"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        close = data["close"]
        volume = data["volume"].fillna(0.0)
        effSpread = data["effSpread"].fillna(0.0)
        lobImb = data["lobImb"].fillna(0.0)

        # Calculate price impact as the signed volume times the effective spread
        price_impact = trade * effSpread

        # Calculate the average price impact over the last 5 bars
        avg_price_impact = ts_mean(price_impact, 5)

        # Normalize by the average volume to get a relative measure
        avg_volume = ts_mean(volume, 5)
        normalized_impact = avg_price_impact / (avg_volume.replace(0, np.nan))

        # Handle NaN values by filling them with 0
        normalized_impact = normalized_impact.fillna(0.0)

        return normalized_impact