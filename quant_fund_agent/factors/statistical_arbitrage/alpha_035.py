"""Alpha#35: Ts_Rank(volume,32) * (1 - Ts_Rank((close+high)-low, 16)) * (1 - Ts_Rank(returns, 32))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import returns, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha035(BaseFactor):
    factor_id = "alpha_035"
    name = "Alpha#35"
    category = "statistical_arbitrage"
    description = (
        "Three-way interaction of time-series ranks: volume rank "
        "(32-day), inverted range rank (16-day), and inverted return "
        "rank (32-day).  Favours high-volume, low-range, low-return "
        "regimes."
    )
    window_length = 32
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ret = returns(data)
        a = ts_rank(data["volume"], 32)
        b = 1.0 - ts_rank((data["close"] + data["high"]) - data["low"], 16)
        c = 1.0 - ts_rank(ret, 32)
        return a * b * c
