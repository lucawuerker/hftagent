"""Alpha#15: -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha015(BaseFactor):
    factor_id = "alpha_015"
    name = "Alpha#15"
    category = "statistical_arbitrage"
    description = (
        "Negative 3-day sum of ranked 3-day correlation between ranked "
        "highs and ranked volume.  Smoothed measure of high-volume "
        "co-movement breakdown."
    )
    window_length = 6
    inputs = ["high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        inner = rank(correlation(rank(data["high"]), rank(data["volume"]), 3))
        return -1.0 * ts_sum(inner, 3)
