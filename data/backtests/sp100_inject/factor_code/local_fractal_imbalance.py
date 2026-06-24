"""Markets often shift from smooth trending behavior to choppy, mean-reverting behavior before larger reversals, which is the intuition behind local fractal/Hurst-based crash diagnostics in the cited paper. When recent price action becomes more anti-persistent relative to a longer window, it suggests crowding, nervous liquidation, or exhaustion of trend-following demand. I would expect the cross-sectional weakest names to be those with the sharpest short-window deterioration in persistence, especially when accompanied by range expansion but limited net progress."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, stddev, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LocalFractalImbalance(BaseFactor):
    """Local fractal imbalance signal from short-vs-long persistence proxies."""

    factor_id = "local_fractal_imbalance"
    name = "Local Fractal Imbalance"
    category = "other"
    description = (
        "Compute a rolling trend-consistency score from close-to-close returns "
        "using short and long windows; the factor is the difference between "
        "short-window and long-window local Hurst-like persistence proxies."
    )
    window_length = 80
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ret_abs = ret.abs().fillna(0.0)

        short_w = 8
        long_w = 40
        eps = 1e-12

        short_mean = ts_mean(ret, short_w)
        long_mean = ts_mean(ret, long_w)

        short_trend = ts_sum(short_mean, short_w) / (ts_sum(ret_abs, short_w) + eps)
        long_trend = ts_sum(long_mean, long_w) / (ts_sum(ret_abs, long_w) + eps)

        short_vol = stddev(ret, short_w)
        long_vol = stddev(ret, long_w)

        short_range = (high - low).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_ratio = ts_mean(short_range, short_w) / (ts_mean(short_range, long_w) + eps)

        persistence_gap = short_trend - long_trend
        volatility_gap = (long_vol - short_vol) / (long_vol.abs() + short_vol.abs() + eps)

        raw = persistence_gap + 0.5 * volatility_gap - 0.25 * (range_ratio - 1.0)
        signal = -rank(raw)

        return signal.reindex(index=close.index, columns=close.columns)
