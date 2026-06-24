"""Market participants tend to react to significant price movements, often leading to overreactions and subsequent corrections. By analyzing the density of zeros in the spectrogram of price movements, we can identify periods of heightened market activity, which may signal impending reversals or continuations, as suggested by the clustering of zeros around specific price levels."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_rank

@register_factor
class SpectrogramZeroDensity(BaseFactor):
    factor_id = "spectrogram_zero_density"
    name = "Spectrogram Zero Density Indicator"
    category = "other"
    description = ("This factor computes the density of zeros in the spectrogram of price movements, normalized by the total number of zeros. "
                   "It captures the local structure of price dynamics, highlighting areas of potential reversal or continuation based on historical price reactions.")
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        price_movement = close.diff().fillna(0.0)
        zero_density = (price_movement == 0).astype(float)
        total_zeros = ts_sum(zero_density, 5)
        density = zero_density / total_zeros.replace(0, np.nan)
        return ts_rank(density, 5)