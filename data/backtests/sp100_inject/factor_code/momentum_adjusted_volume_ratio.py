"""Momentum strategies benefit from the tendency of assets to persist in their price direction due to behavioral biases like herding. By adjusting momentum signals with volume, we can identify trends that are supported by strong trading activity, which may indicate stronger conviction among market participants. This signal computes the ratio of momentum (based on closing prices) to the recent volume, capturing how price movements are supported by trading activity. A higher ratio indicates a strong momentum trend backed by significant volume, while a lower ratio suggests weaker trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, adv, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MomentumAdjustedVolumeRatio(BaseFactor):
    factor_id = "momentum_adjusted_volume_ratio"
    name = "Momentum Adjusted Volume Ratio"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        momentum = delta(close, 5)
        momentum_adjusted = ts_mean(momentum, 20) / adv_volume
        return momentum_adjusted.where(adv_volume > 0, np.nan)