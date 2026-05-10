"""Alpha#44: -1 * correlation(high, rank(volume), 5)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha044(BaseFactor):
    factor_id = "alpha_044"
    name = "Alpha#44"
    category = "statistical_arbitrage"
    description = (
        "Negative 5-day correlation between high price and "
        "cross-sectionally ranked volume.  Fades stocks where highs "
        "and relative volume are tightly co-moving."
    )
    window_length = 5
    inputs = ["high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * correlation(data["high"], rank(data["volume"]), 5)
