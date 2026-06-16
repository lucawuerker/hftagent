"""Volume variation within intraday trading intervals can provide insights into market sentiment and potential price movements. Markets often react to changes in volume, as spikes may indicate heightened interest or the presence of informed traders, which could lead to price changes following a period of low volume. Understanding these dynamics can help traders anticipate short-term price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, delta, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntradayVolumeVariationSignal(BaseFactor):
    factor_id = "intraday_volume_variation_signal"
    name = "Intraday Volume Variation Signal"
    category = "microstructure"
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"]
        avg_volume = ts_mean(volume, 10)
        volume_change = delta(volume, 1) / avg_volume.fillna(1)  # Avoid division by zero
        return volume_change.where(avg_volume > 0, 0.0)  # Mask where avg_volume is zero