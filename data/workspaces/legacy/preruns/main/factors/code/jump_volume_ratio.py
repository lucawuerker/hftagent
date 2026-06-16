"""The presence of sudden price jumps in financial markets is often associated with significant trading volume, indicating strong market sentiment and potential future price movements. According to the paper 'Jump Process Modelling in Finance,' these jumps can signal shifts in market dynamics, making it crucial to monitor the ratio of volume during jumps compared to regular trading periods."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, abs_, sign, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class JumpVolumeRatio(BaseFactor):
    factor_id = "jump_volume_ratio"
    name = "Jump Volume Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of volume during periods of significant price jumps to the average trading volume during non-jump periods."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate price jumps (e.g., 5% threshold)
        price_jump_threshold = 0.05
        price_jump = abs_(delta(close, 1)) > price_jump_threshold

        # Calculate average volume during non-jump periods
        avg_volume_non_jump = ts_mean(volume.where(~price_jump), 60)

        # Calculate volume during jump periods
        volume_during_jumps = volume.where(price_jump)

        # Calculate jump volume ratio
        jump_volume_ratio = ts_sum(volume_during_jumps, 60) / avg_volume_non_jump

        # Replace NaN with 0 for safety
        return jump_volume_ratio.fillna(0.0)