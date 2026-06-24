"""The High-Low Volume Ratio can capture the market sentiment by comparing the trading volume during periods of high price volatility to the volume during periods of low price volatility. High ratios indicate strong conviction in moves, potentially foreshadowing continuation patterns or reversals, as traders react to price fluctuations. This aligns with the behavioral finance theory that suggests traders are likely to take action when prices exhibit significant changes, indicating a heightened sense of urgency or opportunity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, stddev, adv, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HighLowVolumeRatio(BaseFactor):
    factor_id = "high_low_volume_ratio"
    name = "High-Low Volume Ratio"
    category = "volatility"
    inputs = ["high", "low", "volume"]
    description = "This factor computes the ratio of volume during high volatility periods to volume during low volatility periods, identifying stocks that may experience increased trading activity relative to their price movements."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].fillna(0.0)
        low = data["low"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the price volatility as the difference between high and low
        price_volatility = high - low

        # Define high and low volatility thresholds
        high_volatility_mask = price_volatility > ts_mean(price_volatility, 20)
        low_volatility_mask = price_volatility <= ts_mean(price_volatility, 20)

        # Calculate volume during high and low volatility periods
        high_vol_volume = volume.where(high_volatility_mask, 0.0)
        low_vol_volume = volume.where(low_volatility_mask, 0.0)

        # Calculate the average volume during high and low volatility periods
        high_vol_avg = adv(high_vol_volume, 20)
        low_vol_avg = adv(low_vol_volume, 20)

        # Calculate the High-Low Volume Ratio
        ratio = high_vol_avg / low_vol_avg.replace(0, np.nan)

        return ratio