"""When price moves strongly but closes near the open despite elevated volume, it often signals that aggressive order flow was absorbed by passive liquidity rather than establishing a durable trend. This is especially plausible in short-horizon markets where informed or urgent trading is temporary and market makers fade extremes; the behavior aligns with the paper's emphasis on conflict resolution and regulation as a metaphor for absorbing pressure rather than amplifying it. Cross-sectionally, names showing repeated high-volume rejection of intrabar extremes should underperform recent direction and mean-revert as the imbalance dissipates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, correlation, delta, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeAbsorptionPressure(BaseFactor):
    """Measures large-range, high-volume bars that close near the open."""

    factor_id = "volume_absorption_pressure"
    name = "Volume Absorption Pressure"
    category = "microstructure"
    description = (
        "Measures the degree to which a bar's range and volume are large while "
        "the close remains near the open, normalized so that stronger absorption "
        "gets a higher score."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        true_range = (high - low).abs().fillna(0.0)
        body = (close - open_).abs().fillna(0.0)

        range_safe = true_range.replace(0.0, np.nan)
        close_near_open = (1.0 - (body / range_safe)).clip(lower=0.0, upper=1.0).fillna(0.0)

        vol_rel = volume / adv(volume, 20)
        vol_rel = vol_rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_pressure = rank(vol_rel)

        range_pressure = rank(true_range / ts_mean(true_range.replace(0.0, np.nan), 20).replace(0.0, np.nan))
        range_pressure = range_pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        absorption = close_near_open * range_pressure * volume_pressure
        absorption = absorption.fillna(0.0)

        # Emphasize repeated rejection of intrabar extremes and penalize persistent direction.
        recent_rejection = ts_rank(abs_(body / range_safe), 10)
        recent_rejection = recent_rejection.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        directional_bias = ts_rank(delta(close, 5).abs(), 10)
        directional_bias = directional_bias.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        mean_reversion_component = (1.0 - recent_rejection) * (1.0 - directional_bias)
        mean_reversion_component = mean_reversion_component.fillna(0.0)

        signal = absorption * (1.0 + mean_reversion_component)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return signal
