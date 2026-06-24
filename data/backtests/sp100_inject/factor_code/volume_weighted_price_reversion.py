"""As observed in market behavior, significant price changes often revert to their mean over time, especially when accompanied by high trading volumes. High volume can indicate strong conviction behind price movements, but if prices deviate significantly from their volume-weighted average, it may signal an overreaction in either direction, creating a mean-reversion opportunity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import vwap, delta, ts_mean, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeWeightedPriceReversion(BaseFactor):
    factor_id = "volume_weighted_price_reversion"
    name = "Volume Weighted Price Reversion"
    category = "mean_reversion"
    description = "This signal computes the difference between the current price and the volume-weighted average price over a specified lookback period. A high positive or negative deviation indicates potential mean-reversion opportunities."
    window_length = 60
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        vwap_value = vwap(data)
        price_deviation = close - vwap_value
        mean_deviation = ts_mean(price_deviation, 60)
        return ts_rank(price_deviation, 60) - mean_deviation