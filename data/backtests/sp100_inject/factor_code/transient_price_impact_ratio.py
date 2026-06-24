"""In high-frequency trading environments, the transient price impact of trades can reveal the underlying sentiment of market participants. This factor leverages the relationship between volume and price changes to gauge whether buying or selling pressure is dominant, which can indicate future price movements. The insights from Luo and Schied (2018) on risk-averse investors and transient price impact suggest that understanding market reactions to trades can lead to profitable strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, delta, vwap


@register_factor
class TransientPriceImpactRatio(BaseFactor):
    factor_id = "transient_price_impact_ratio"
    name = "Transient Price Impact Ratio"
    category = "microstructure"
    description = "This signal computes the ratio of price changes to the volume traded over a specific period, capturing the transient impact of trades on price. It provides a measure of how much price is affected by trading activity, indicating whether the market is currently experiencing buying or selling pressure."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        price_change = delta(close, 1)
        total_volume = ts_sum(volume, 5)

        # Avoid division by zero by replacing zeros in total_volume with NaN
        total_volume = total_volume.replace(0, np.nan)

        # Calculate the transient price impact ratio
        impact_ratio = price_change / total_volume

        return impact_ratio