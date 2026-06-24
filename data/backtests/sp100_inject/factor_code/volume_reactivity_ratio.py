"""Market behavior often shows that sudden shifts in trading volume can lead to price changes, especially when they deviate from historical norms. By calculating the ratio of current volume to an average volume over a defined period, this factor aims to capture the potential for price movement following unusual trading activity, leveraging the idea that increased volume can indicate market sentiment shifts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, ts_mean


@register_factor
class VolumeReactivityRatio(BaseFactor):
    factor_id = "volume_reactivity_ratio"
    name = "Volume Reactivity Ratio"
    category = "microstructure"
    inputs = ["volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        avg_volume = adv(volume, 20)
        ratio = volume / avg_volume.replace(0, np.nan)
        return ratio