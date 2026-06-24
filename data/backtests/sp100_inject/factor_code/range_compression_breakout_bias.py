"""When a bar closes near its extreme after trading inside a very narrow range relative to its recent history, it often signals latent order imbalance rather than equilibrium. The averaging-principle perspective in the cited paper suggests that fast fluctuations can average out, while a persistent directional drift remains in the slow component; here we try to isolate that slow drift after a temporary compression phase. This should complement existing range- and reversal-style signals by targeting continuation only when compression resolves with strong directional conviction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeCompressionBreakoutBias(BaseFactor):
    """Range compression and close-location breakout continuation bias."""

    factor_id = "range_compression_breakout_bias"
    name = "Range Compression Breakout Bias"
    category = "momentum"
    description = (
        "Measures the interaction of short-horizon range compression and "
        "close location within the bar. A positive score is high when "
        "today's true range is small versus recent average, but the close is "
        "near the high; a negative score is the mirror image near the low."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        true_range = (high - low).abs()
        avg_range = ts_mean(true_range, 10)
        compression = (avg_range / true_range.replace(0, np.nan)) - 1.0
        compression = compression.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        compression = compression.clip(lower=-5.0, upper=5.0)

        bar_span = (high - low).replace(0, np.nan)
        close_location = ((close - low) / bar_span) * 2.0 - 1.0
        close_location = close_location.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        close_location = close_location.clip(lower=-1.0, upper=1.0)

        volume_pressure = volume / adv(volume, 20)
        volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        volume_pressure = volume_pressure.clip(lower=0.25, upper=4.0)

        breakout_bias = compression * close_location * volume_pressure

        confirmation = ((close - ts_mean(close, 5)) / ts_mean(abs_(close), 5).replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
        confirmation = confirmation.fillna(0.0).clip(lower=-3.0, upper=3.0)

        return (breakout_bias + 0.5 * confirmation).fillna(0.0)
