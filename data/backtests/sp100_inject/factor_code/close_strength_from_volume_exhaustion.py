"""When a bar closes near its high or low on unusually low volume, it often reflects a one-sided price move that was not strongly confirmed by participation. In line with the order-book intuition in Maslov & Mills (2001), thin participation makes the apparent directional move more fragile because there is less evidence of persistent supply-demand imbalance behind it. Conversely, a strong close on elevated volume is more likely to reflect genuine imbalance and should be less prone to immediate reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseStrengthFromVolumeExhaustion(BaseFactor):
    """Close-location strength downweighted by relative volume exhaustion."""

    factor_id = "close_strength_from_volume_exhaustion"
    name = "Close Strength from Volume Exhaustion"
    category = "microstructure"
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).replace(0, np.nan)
        close_location = ((close - low) / bar_range) * 2.0 - 1.0
        close_location = close_location.fillna(0.0)

        vol20 = adv(volume, 20)
        vol_norm = volume / vol20.replace(0, np.nan)
        vol_norm = vol_norm.fillna(1.0)

        vol_exhaustion = (2.0 - vol_norm).clip(lower=0.0, upper=2.0)
        vol_exhaustion = rank(-vol_exhaustion).fillna(0.5)

        smooth_close = ts_mean(close_location, 3).fillna(close_location)
        signal = smooth_close * (2.0 * vol_exhaustion - 1.0)

        return signal.reindex(index=close.index, columns=close.columns)
