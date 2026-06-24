"""Momentum strategies often rely on price movements, but incorporating volume can enhance signal quality. Stocks exhibiting momentum alongside higher trading volumes may indicate stronger conviction in price movements, leading to higher probabilities of continuation. This aligns with behavioral finance theories, which suggest that higher volume often correlates with increased investor interest and commitment to price trends."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, adv, ts_mean)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeWeightedMomentum(BaseFactor):
    factor_id = "volume_weighted_momentum"
    name = "Volume Weighted Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        price_change = delta(close, 5)
        momentum_signal = ts_sum(price_change * volume, 5)
        return momentum_signal.where(volume > adv_volume, 0.0)