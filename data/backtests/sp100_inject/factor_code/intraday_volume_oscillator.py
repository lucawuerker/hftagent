"""Intraday volume often exhibits cyclical patterns throughout the trading day, characterized by peaks and troughs. By measuring the momentum of volume changes relative to historical averages, traders can identify periods of heightened activity that may signal upcoming price movements, akin to momentum strategies in price action. This aligns with findings in market microstructure literature that suggest increased volume can precede price changes."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, delta, adv, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayVolumeOscillator(BaseFactor):
    factor_id = "intraday_volume_oscillator"
    name = "Intraday Volume Oscillator"
    category = "momentum"
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        adv5 = adv(volume, 5)
        volume_change = delta(volume, 1)
        oscillator = volume_change / adv5
        return oscillator.where(adv5 != 0, 0.0)