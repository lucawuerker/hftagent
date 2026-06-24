"""Price momentum is a well-documented phenomenon where assets that have performed well in the past continue to do so in the near future. By incorporating volume as a weight in this momentum calculation, we can enhance the signal's reliability; higher volume often indicates stronger conviction among market participants (as noted in the research on momentum strategies). This factor seeks to exploit the persistence of price trends while adjusting for the confidence signaled by trading volume."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_mean, adv, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceMomentumWithVolumeAdjustment(BaseFactor):
    factor_id = "price_momentum_with_volume_adjustment"
    name = "Price Momentum with Volume Adjustment"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        momentum = delta(close, 5)
        momentum_adjusted = momentum * (volume / adv_volume)
        return ts_rank(momentum_adjusted, 60)