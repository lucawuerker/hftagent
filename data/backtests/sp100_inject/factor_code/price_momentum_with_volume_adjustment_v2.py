"""Price momentum strategies often overlook the importance of volume, which can signal the strength of price movements. By incorporating volume into the momentum calculation, we can better assess whether price changes are supported by significant trading activity, thereby improving the robustness of momentum signals and reducing false signals. This approach builds on the idea that higher volume during price increases indicates stronger buying interest, suggesting a continuation of that trend."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, adv, ts_sum, ts_mean, sign)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceMomentumWithVolumeAdjustmentV2(BaseFactor):
    factor_id = "price_momentum_with_volume_adjustment_v2"
    name = "Volume-Adjusted Price Momentum"
    category = "momentum"
    description = "This factor computes the momentum of price changes adjusted for volume, capturing the relationship between price movements and trading volume to identify potential trends. It is calculated as the price change over a defined period, weighted by the corresponding volume changes during that period."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        price_change = delta(close, 5)
        weighted_momentum = price_change * (volume / adv_volume)
        return weighted_momentum.where(adv_volume > 0, 0.0)