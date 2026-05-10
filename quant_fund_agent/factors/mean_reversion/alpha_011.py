"""Alpha#11: (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, ts_max, ts_min, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha011(BaseFactor):
    factor_id = "alpha_011"
    name = "Alpha#11"
    category = "mean_reversion"
    description = (
        "Interaction of the 3-day max and min of (VWAP − close) with "
        "ranked 3-day volume change.  Captures mean-reversion around "
        "VWAP amplified by volume acceleration."
    )
    window_length = 3
    inputs = ["close", "volume", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        spread = vwap(data) - data["close"]
        lhs = rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))
        rhs = rank(delta(data["volume"], 3))
        return lhs * rhs
