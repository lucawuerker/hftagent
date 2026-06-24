"""Close-Weighted Trend Persistence: When a bar closes near its high after an up bar, it often signals persistent buyer initiative rather than transient noise; conversely, closes near the low after a down bar imply sustained selling pressure. This factor emphasizes directional continuation only when the close is strong relative to the full range and recent returns are already aligned, which helps avoid chasing weak or noisy moves. The idea is grounded in the broader price-action literature and complements the existing range/close-pressure family by focusing on persistence of directional conviction rather than pure reversal or compression effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, returns, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseWeightedTrendPersistence(BaseFactor):
    """Close-weighted continuation signal from aligned returns and range position."""

    factor_id = "close_weighted_trend_persistence"
    name = "Close-Weighted Trend Persistence"
    category = "momentum"
    description = (
        "Compute a signed momentum score from recent returns, then amplify it "
        "by the bar’s close location within the high-low range so that trend "
        "bars closing near extremes receive the highest values. Smooth or "
        "aggregate the score over a short lookback to produce a cross-sectional "
        "signal by ticker."
    )
    window_length = 10
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        prev_close = close.shift(1)
        bar_return = returns(data).fillna(0.0)
        trend_lookback = ts_mean(bar_return, 5).fillna(0.0)

        range_ = (high - low).replace(0, np.nan)
        close_location = ((close - low) / range_).clip(0.0, 1.0)
        close_location = close_location.fillna(0.5)
        close_pressure = (close_location - 0.5) * 2.0

        direction = np.sign(prev_close.diff()).replace(0, np.nan)
        direction = direction.fillna(np.sign(bar_return)).fillna(0.0)

        aligned_momentum = trend_lookback * direction
        conviction = aligned_momentum.where(aligned_momentum.abs() > 0.0, 0.0)
        conviction = conviction * (1.0 + close_pressure.abs())
        conviction = conviction * close_pressure

        persistence = ts_mean(conviction, 3).fillna(0.0)
        breakout_bonus = rank(delta(close, 2).fillna(0.0))

        signal = persistence + 0.25 * breakout_bonus * close_pressure
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
