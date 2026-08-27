"""Alpha#45: -1 * (rank(sum(delay(close,5),20)/20) * correlation(close,volume,2) * rank(correlation(sum(close,5),sum(close,20),2)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delay, rank, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha045(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_045"
    name = "Alpha#45"
    category = "statistical_arbitrage"
    description = (
        "Negative product of ranked 20-day mean of 5-day-lagged close, "
        "2-day close-volume correlation, and ranked 2-day correlation "
        "between short (5-day) and medium (20-day) close sums."
    )
    window_length = 25
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        a = rank(ts_mean(delay(close, 5), 20))
        b = correlation(close, volume, 2)
        c = rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2))
        return -1.0 * a * b * c
