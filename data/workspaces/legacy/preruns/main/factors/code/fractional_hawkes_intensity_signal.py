"""The fractional Hawkes process models the self-exciting nature of market events, where past trades influence future trading activity. This behavior can create momentum in price movements, particularly after clusters of trades, as market participants react to perceived trends. By analyzing the intensity of these self-exciting processes with a Mittag-Leffler kernel, we can capture long-memory effects and anticipate price movements more accurately than with traditional models.

This signal computes the conditional intensity of trades using a fractional Hawkes process, which incorporates the history of trading events to predict future trading behavior. It provides insights into how recent trading activity influences price movements over time.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, ts_max, ts_argmax, delay, stddev, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class FractionalHawkesIntensitySignal(BaseFactor):
    factor_id = "fractional_hawkes_intensity_signal"
    name = "Fractional Hawkes Intensity Signal"
    category = "microstructure"
    inputs = ["trade", "volume", "effSpread"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        eff_spread = data["effSpread"].replace(0, np.nan)

        # Calculate the intensity based on the trade history
        intensity = ts_sum(trade, 10) / ts_mean(volume, 10)
        intensity = intensity.where(eff_spread.notna(), 0.0)

        # Normalize the intensity
        intensity = intensity / stddev(intensity, 60)
        intensity = intensity.fillna(0.0)

        return intensity