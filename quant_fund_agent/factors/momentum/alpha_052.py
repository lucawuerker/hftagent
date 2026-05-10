"""Alpha#52: ((-1*ts_min(low,5) + delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/220)) * ts_rank(volume,5)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, rank, returns, ts_min, ts_rank, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha052(BaseFactor):
    factor_id = "alpha_052"
    name = "Alpha#52"
    category = "momentum"
    description = (
        "5-day low support change times ranked long-vs-short return "
        "differential (240-day minus 20-day) times 5-day volume "
        "ts-rank.  Momentum signal combining support-level shifts, "
        "trend, and volume."
    )
    window_length = 245
    inputs = ["close", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        low = data["low"]
        vol = data["volume"]
        ret = returns(data)

        low5 = ts_min(low, 5)
        support_shift = -1.0 * low5 + delay(low5, 5)
        trend = rank((ts_sum(ret, 240) - ts_sum(ret, 20)) / 220.0)
        vol_rank = ts_rank(vol, 5)
        return support_shift * trend * vol_rank
