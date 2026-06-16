"""This factor leverages the relationship between intraday trading volume and the bid-ask spread. In highly liquid markets, a lower spread relative to volume signals strong demand and can indicate bullish sentiment, while a higher spread relative to volume may signal bearish conditions due to weaker demand. This insight aligns with findings in market microstructure literature where spreads influence price movements and trading costs."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, abs_, delay


@register_factor
class IntradayVolumeSpreadRatio(BaseFactor):
    factor_id = "intraday_volume_spread_ratio"
    name = "Intraday Volume Spread Ratio"
    category = "microstructure"
    inputs = ["volume", "spread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        spread = data["spread"].replace(0, np.nan)

        # Calculate the sum of volume over the day
        total_volume = ts_sum(volume, 2340)  # Assuming 2340 bars in a trading day

        # Calculate the mean spread over the day
        mean_spread = ts_mean(spread, 2340)

        # Calculate the ratio of total volume to mean spread
        ratio = total_volume / mean_spread

        # Handle NaN values in the ratio
        ratio = ratio.fillna(0.0)

        return ratio