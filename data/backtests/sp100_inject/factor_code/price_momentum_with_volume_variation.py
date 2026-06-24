"""Stocks that exhibit significant price momentum paired with an increase in trading volume may indicate strong underlying trends and investor interest. This aligns with behavioral finance theories suggesting that increased trading activity can amplify price changes and signal continued momentum, as outlined in the work on momentum strategies by Jegadeesh and Titman (1993)."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, adv, correlation)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceMomentumWithVolumeVariation(BaseFactor):
    factor_id = "price_momentum_with_volume_variation"
    name = "Price Momentum with Volume Variation"
    category = "momentum"
    description = "This factor computes the price momentum of a stock over a specified period and correlates it with the percentage change in trading volume. A positive correlation between rising prices and increasing volumes is expected to predict further price increases."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        momentum = delta(close, 5)  # 5-bar price momentum
        volume_change = delta(volume, 5)  # 5-bar volume change
        correlation_signal = correlation(momentum, volume_change, 5)  # Correlation over 5 bars
        return correlation_signal.where(adv20 < volume, 0.0)  # Fallback to 0.0 if volume is below ADV
