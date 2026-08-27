"""Compressed-depth acceptance drift.

Mechanism: a sequence of narrow ranges can reflect quiet consumption of displayed
and latent liquidity by a scheduled trader.  When participation subsequently
expands and the close is accepted near one end of the range, the depleted side
of the book is likely to permit further directional movement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CompressedDepthAcceptanceDrift(BaseFactor):
    """Directional acceptance after a causal estimate of range compression."""

    factor_id = "compressed_depth_acceptance_drift"
    name = "Compressed-Depth Acceptance Drift"
    category = "microstructure"
    description = (
        "Cross-sectional rank of directional close acceptance on abnormal "
        "volume, activated only after the name's prior ranges were compressed "
        "relative to its own trailing range."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float)
        high = data["high"].astype(float).fillna(close)
        low = data["low"].astype(float).fillna(close)
        volume = data["volume"].astype(float).fillna(0.0).clip(lower=0.0)

        bar_high = pd.DataFrame(
            np.maximum(high.to_numpy(), close.to_numpy()),
            index=close.index,
            columns=close.columns,
        )
        bar_low = pd.DataFrame(
            np.minimum(low.to_numpy(), close.to_numpy()),
            index=close.index,
            columns=close.columns,
        )
        bar_range = bar_high - bar_low
        safe_range = bar_range.where(bar_range > 1e-12)

        close_location = (
            2.0 * (close - bar_low) / safe_range - 1.0
        ).clip(lower=-1.0, upper=1.0).fillna(0.0)

        log_volume = np.log1p(volume)
        volume_mean = log_volume.rolling(20, min_periods=10).mean()
        volume_std = log_volume.rolling(20, min_periods=10).std()
        volume_surprise = (
            (log_volume - volume_mean) / volume_std.where(volume_std > 1e-12)
        ).clip(lower=0.0, upper=3.0).fillna(0.0)

        relative_range = bar_range / close.abs().where(close.abs() > 1e-12)
        prior_short_range = relative_range.shift(1).rolling(5, min_periods=5).mean()
        prior_long_range = relative_range.shift(1).rolling(20, min_periods=15).mean()
        compression = (
            1.0 - prior_short_range / prior_long_range.where(prior_long_range > 1e-12)
        ).clip(lower=0.0, upper=1.0).fillna(0.0)

        acceptance_pressure = close_location * volume_surprise * compression
        return rank(acceptance_pressure).fillna(0.5)
