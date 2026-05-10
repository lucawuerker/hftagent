"""Alpha#32: scale((sum(close,7)/7) - close) + 20*scale(correlation(vwap, delay(close,5), 230))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delay, scale, ts_mean, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha032(BaseFactor):
    factor_id = "alpha_032"
    name = "Alpha#32"
    category = "mean_reversion"
    description = (
        "Scaled 7-day MA deviation from close (short-term reversion) "
        "plus 20x-scaled 230-day correlation of VWAP with the 5-day "
        "lagged close (long-horizon fair-value anchor)."
    )
    window_length = 235
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        short_term = scale(ts_mean(close, 7) - close)
        long_term = 20.0 * scale(correlation(v, delay(close, 5), 230))
        return short_term + long_term
