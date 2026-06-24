"""Markets often persist in regime-like bullish or bearish states rather than switching on every noisy bar, consistent with the hierarchical regime logic discussed in the HHMM paper. A trend that closes near its bar high after already maintaining a positive multi-bar drift suggests both directional persistence and low intrabar selling pressure, which should outperform a simple return-only momentum signal by filtering out weak or noisy moves. This factor aims to capture the market's current 'conviction' in the prevailing trend, especially when price action is trending and bars are not meaningfully retracing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TrendStateConviction(BaseFactor):
    """Trend-state conviction from persistent drift and close location within the bar."""

    factor_id = "trend_state_conviction"
    name = "Trend State Conviction"
    category = "momentum"
    description = (
        "Compute a rolling z-scored trend strength from recent close-to-close "
        "returns and multiply it by a normalized close position within the bar. "
        "The signal is positive when a stock has persistent upside drift and "
        "closes near the high of the bar, and negative for persistent downside "
        "drift with closes near the low."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        close_ret = delta(close, 1) / close.shift(1).replace(0, np.nan)
        drift_5 = ts_sum(close_ret, 5)
        drift_20_mean = ts_mean(close_ret, 20)
        drift_20_std = close_ret.rolling(20, min_periods=20).std()
        trend_strength = (drift_20_mean + drift_5) / (drift_20_std + 1e-8)

        bar_range = (high - low).replace(0, np.nan)
        close_pos = (2.0 * (close - low) / bar_range) - 1.0
        close_pos = close_pos.fillna(0.0)

        persistence = rank(abs_(ts_sum(close_ret, 10)))
        conviction = trend_strength * close_pos * persistence

        return conviction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
