"""The paper’s core message is that complex dynamics are often easier to forecast in spectral space than in raw time series, because low-frequency structure captures persistent state while high-frequency modes capture transient shocks. For bars, a similar idea is that when most of a bar’s action is concentrated in the body and the close continues to move away from the open over several bars, trend state is more stable; when the high-low range is dominated by intrabar excursions but the close barely advances, the move is more likely to be noise or exhaustion. This factor measures the drift in “range energy” through a smoothed combination of bar body and directional range efficiency, favoring names whose recent price action is becoming increasingly coherent rather than merely volatile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import decay_linear, delta, rank, ts_mean, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class SpectralRangeEnergyDrift(BaseFactor):
    """Rolling z-scored range-energy drift with short-horizon smoothing."""

    factor_id = "spectral_range_energy_drift"
    name = "Spectral Range Energy Drift"
    category = "other"
    description = (
        "Compute a rolling z-scored signal from the difference between absolute "
        "close-to-open body size and total high-low range, then smooth it over a "
        "short horizon. Positive values indicate persistent directional absorption "
        "of range into the close; negative values indicate unstable, oscillatory "
        "movement with poor directional conversion."
    )
    window_length = 30
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].replace(0, np.nan)
        high = data["high"]
        low = data["low"]
        close = data["close"]

        body = (close - open_).abs()
        range_ = (high - low).abs().replace(0, np.nan)

        body_efficiency = body / range_
        direction = delta(close, 1).abs()
        direction_efficiency = direction / range_

        # A smooth proxy for spectral coherence: more body capture and directional
        # conversion relative to raw intrabar range implies more stable structure.
        energy = body_efficiency - (1.0 - direction_efficiency)
        energy = energy.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        fast = ts_mean(energy, 5)
        slow = ts_mean(energy, 20)
        drift = fast - slow

        drift_mean = ts_mean(drift, 20)
        drift_std = drift.rolling(20, min_periods=20).std()
        z = (drift - drift_mean) / drift_std.replace(0.0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        smoothed = decay_linear(z, 5)
        return smoothed.fillna(0.0)
