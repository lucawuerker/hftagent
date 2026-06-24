"""The intensity of Hawkes processes is known to increase with past activity, suggesting that strong buying or selling pressure can lead to further price movements in the same direction. By analyzing the inverse of the intensity ratio, we can identify potential market reversals when the intensity of buying decreases relative to the selling intensity, signaling a shift in momentum."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class InverseHawkesIntensityRatio(BaseFactor):
    factor_id = "inverse_hawkes_intensity_ratio"
    name = "Inverse Hawkes Intensity Ratio"
    category = "momentum"
    inputs = ["volume", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"].fillna(0.0)
        close = data["close"]

        buying_intensity = ts_sum(volume.where(close.diff() > 0, 0), 5)
        selling_intensity = ts_sum(volume.where(close.diff() < 0, 0), 5)

        # Avoid division by zero
        intensity_ratio = buying_intensity / selling_intensity.replace(0, np.nan)
        inverse_intensity_ratio = 1 / intensity_ratio

        return inverse_intensity_ratio
