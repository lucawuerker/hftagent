"""Alpha#50: -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank, ts_max, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha050(BaseFactor):
    factor_id = "alpha_050"
    name = "Alpha#50"
    category = "statistical_arbitrage"
    description = (
        "Negative 5-day max of the ranked 5-day correlation between "
        "ranked volume and ranked VWAP.  Fades peak volume-VWAP "
        "co-movement."
    )
    window_length = 10
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        inner = rank(correlation(rank(data["volume"]), rank(v), 5))
        return -1.0 * ts_max(inner, 5)
