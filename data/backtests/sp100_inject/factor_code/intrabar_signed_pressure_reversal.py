"""When a bar closes near its extreme on unusually high volume, it often reflects aggressive liquidity-taking that temporarily pushes price away from a short-horizon equilibrium. If that pressure is not supported by strong follow-through in the next bars, the move tends to fade as inventory rebalances and the market digests the impact; this is especially plausible in noisy, high-frequency settings where transient price impact dominates. The idea complements momentum-style seeds by explicitly targeting exhaustion after one-sided intrabar pressure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delta, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarSignedPressureReversal(BaseFactor):
    """Mean-reversion signal from signed intrabar pressure, volume surprise, and range expansion."""

    factor_id = "intrabar_signed_pressure_reversal"
    name = "Intrabar Signed Pressure Reversal"
    category = "mean_reversion"
    description = (
        "Measure signed intrabar pressure from close location within the bar, "
        "then scale it by volume surprise and recent range expansion."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        close_loc = ((close - open_) / bar_range).clip(-1.0, 1.0)
        close_loc = close_loc.fillna(0.0)

        signed_pressure = -close_loc

        vol_adv = adv(volume, 20)
        volume_surprise = (volume / vol_adv.replace(0.0, np.nan)) - 1.0
        volume_surprise = volume_surprise.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        range_ = (high - low).fillna(0.0)
        range_mean = ts_mean(range_, 10)
        range_expansion = (range_ / range_mean.replace(0.0, np.nan)) - 1.0
        range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        intrabar_move = close - open_
        recent_direction = ts_sum(delta(close, 1), 3).fillna(0.0)
        exhaustion = -np.sign(intrabar_move) * np.abs(recent_direction).clip(lower=0.0)

        signal = signed_pressure * (1.0 + volume_surprise) * (1.0 + range_expansion)
        signal = signal + 0.5 * exhaustion

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
