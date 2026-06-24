"""The volatility of an asset is often mean-reverting; thus, deviations from its historical average can signal potential reversals. By calculating the spread between an asset's realized volatility over a recent period and its long-term average, traders can identify overbought or oversold conditions. This approach aligns with the concept that volatility tends to revert to its mean, as discussed in the literature surrounding stochastic volatility models."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HistoricalRealizedVolatilitySpread(BaseFactor):
    factor_id = "historical_realized_volatility_spread"
    name = "Historical Realized Volatility Spread"
    category = "volatility"
    description = "This signal computes the difference between the historical realized volatility for a specified short-term period and the long-term average realized volatility."
    window_length = 60
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        short_term_vol = stddev(close, 20)
        long_term_vol = ts_mean(stddev(close, 60), 60)
        spread = short_term_vol - long_term_vol
        return spread.fillna(0.0)