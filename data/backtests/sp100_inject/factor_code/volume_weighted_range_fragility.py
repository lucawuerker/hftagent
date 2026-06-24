"""When most of a bar’s volume occurs while price is still near one edge of the range, the move is often fragile: the market has revealed directional intent without achieving broad two-sided acceptance. If the bar then closes well away from the volume-weighted center of its own range, it can indicate an overcrowded micro-move that is prone to short-horizon reversal or continuation failure, depending on the sign of the imbalance. This complements existing range and wick signals by explicitly conditioning price location on traded participation rather than only on OHLC geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeWeightedRangeFragility(BaseFactor):
    """Fragility signal from close displacement versus volume-weighted range location."""

    factor_id = "volume_weighted_range_fragility"
    name = "Volume-Weighted Range Fragility"
    category = "microstructure"
    description = (
        "Scores how far the close sits from the bar's volume-weighted range "
        "center, amplified when participation is heavy relative to recent volume."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0, np.nan)
        mid = (high + low) * 0.5

        close_pos = ((close - low) / rng).clip(0.0, 1.0)
        open_pos = ((open_ - low) / rng).clip(0.0, 1.0)

        # Proxy for where traded volume likely concentrated within the bar:
        # if the close is near an edge but the open is near the opposite edge,
        # the bar is more likely to have traded through a directional imbalance.
        edge_pressure = (close_pos - 0.5) - 0.5 * (close_pos - open_pos)
        volume_center_proxy = mid + edge_pressure * rng
        displacement = (close - volume_center_proxy) / rng

        # Emphasize unusual participation using recent average volume.
        vol_norm = volume / adv(volume, 20)
        vol_norm = vol_norm.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        vol_intensity = rank(vol_norm)

        # Smooth the raw displacement a bit to avoid single-bar noise.
        fragility_core = ts_mean(displacement, 3)
        fragility_core = fragility_core.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        signal = fragility_core * (1.0 + vol_intensity) * (volume / adv(volume, 20)).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
