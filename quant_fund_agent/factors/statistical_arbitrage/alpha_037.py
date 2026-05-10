"""Alpha#37: rank(correlation(delay(open - close, 1), close, 200)) + rank(open - close)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delay, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha037(BaseFactor):
    factor_id = "alpha_037"
    name = "Alpha#37"
    category = "statistical_arbitrage"
    description = (
        "Ranked 200-day correlation between the lagged intraday gap "
        "and close, plus ranked current gap.  Long-horizon structural "
        "relationship between overnight gap persistence and price."
    )
    window_length = 201
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        gap = data["open"] - data["close"]
        a = rank(correlation(delay(gap, 1), data["close"], 200))
        b = rank(gap)
        return a + b
