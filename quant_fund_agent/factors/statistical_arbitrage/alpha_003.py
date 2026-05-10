"""Alpha#3: -1 * correlation(rank(open), rank(volume), 10)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha003(BaseFactor):
    factor_id = "alpha_003"
    name = "Alpha#3"
    category = "statistical_arbitrage"
    description = (
        "Negative 10-day rolling correlation between cross-sectionally "
        "ranked open price and ranked volume."
    )
    window_length = 10
    inputs = ["open", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * correlation(rank(data["open"]), rank(data["volume"]), 10)
