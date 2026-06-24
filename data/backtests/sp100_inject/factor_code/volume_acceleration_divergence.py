"""When price advances but traded volume fails to accelerate, the move is often supported by weak participation and is more likely to stall or reverse. Conversely, downside moves accompanied by rising volume can indicate forced liquidation and trend persistence. This factor is inspired by the robust aggregation idea in the paper: we try to distinguish the 'consensus' price move from noisy or corrupt-like bursts by focusing on whether volume confirms the bar-to-bar price impulse."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, sign, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeAccelerationDivergence(BaseFactor):
    """Cross-sectional divergence between price impulse and volume acceleration."""

    factor_id = "volume_acceleration_divergence"
    name = "Volume Acceleration Divergence"
    category = "other"
    description = (
        "Measures the divergence between short-term price momentum and short-term "
        "volume acceleration, then cross-sectionally ranks stocks where price is "
        "moving faster than volume participation (or vice versa). High values "
        "indicate bullish price change with weak volume confirmation; low values "
        "indicate bearish pressure with strong volume confirmation."
    )
    window_length = 10
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]

        # Short-term price momentum and volume acceleration.
        price_mom = delta(close, 3)
        volume_acc = delta(volume, 3)

        # Normalize by recent activity to compare impulse versus participation.
        price_scale = ts_mean(abs_(delta(close, 1)), 5).replace(0, np.nan)
        volume_scale = ts_mean(abs_(delta(volume, 1)), 5).replace(0, np.nan)

        norm_price_mom = price_mom / price_scale
        norm_volume_acc = volume_acc / volume_scale

        # Positive when price is moving up more than volume is accelerating.
        divergence = norm_price_mom - norm_volume_acc

        # Emphasize directional interpretation: bullish divergence positive,
        # bearish liquidation pressure negative.
        directional = sign(price_mom) * abs_(divergence)

        # Cross-sectional ranking for each bar; output stays numeric and aligned.
        out = rank(directional)
        return out
