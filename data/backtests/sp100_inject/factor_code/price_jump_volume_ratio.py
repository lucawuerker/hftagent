"""The relationship between the size of price jumps and the volume traded during those jumps can indicate market sentiment and potential future price movements. A higher ratio suggests strong conviction behind the price change, indicating that the market may continue in that direction. This idea is supported by findings from market microstructure models that emphasize the importance of volume in confirming price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, abs_, adv, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PriceJumpVolumeRatio(BaseFactor):
    factor_id = "price_jump_volume_ratio"
    name = "Price Jump to Volume Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of the absolute price change (jump) to the corresponding volume traded during that price change, across a specified time interval."
    inputs = ["close", "volume"]
    window_length = 5

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        price_jump = abs_(delta(close, 1))  # Absolute price change
        volume_avg = adv(volume, self.window_length)  # Average volume over the window

        # Calculate the ratio of price jump to average volume
        ratio = price_jump / volume_avg.where(volume_avg != 0, np.nan)  # Avoid division by zero

        return ratio