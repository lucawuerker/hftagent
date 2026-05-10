"""Alpha#26: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, ts_max, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha026(BaseFactor):
    factor_id = "alpha_026"
    name = "Alpha#26"
    category = "statistical_arbitrage"
    description = (
        "Negative 3-day max of the 5-day correlation between "
        "time-series ranked volume and time-series ranked high.  "
        "Fades peak volume-high co-movement."
    )
    window_length = 13
    inputs = ["high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        vol_rank = ts_rank(data["volume"], 5)
        high_rank = ts_rank(data["high"], 5)
        return -1.0 * ts_max(correlation(vol_rank, high_rank, 5), 3)
