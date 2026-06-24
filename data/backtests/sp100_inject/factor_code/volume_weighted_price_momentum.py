"""Higher trading volume often signals stronger conviction behind price movements. By weighting price momentum by volume, we can capture trends that are supported by strong market participation, which is generally more sustainable. This approach aligns with the principles of behavioral finance, where increased volume indicates investor confidence and can lead to continued price movement in the same direction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (delta, ts_sum, returns, adv, ts_mean)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeWeightedPriceMomentum(BaseFactor):
    factor_id = "volume_weighted_price_momentum"
    name = "Volume Weighted Price Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv_volume = adv(volume, 20)
        price_returns = returns(data)
        weighted_momentum = price_returns * volume
        momentum_signal = ts_mean(weighted_momentum, 20)
        return momentum_signal.where(volume > adv_volume, 0.0)