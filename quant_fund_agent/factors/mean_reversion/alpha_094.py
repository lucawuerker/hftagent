"""Alpha#94: (rank(vwap - ts_min(vwap, 12)) ^ Ts_Rank(corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)) * -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_min, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha094(BaseFactor):
    factor_id = "alpha_094"
    name = "Alpha#94"
    category = "mean_reversion"
    description = (
        "Negative ranked VWAP distance from 12-day low raised to "
        "the power of ts-ranked VWAP-ts-rank vs ADV60-ts-rank "
        "correlation.  Mean-reversion around the VWAP floor."
    )
    window_length = 80
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        volume = data["volume"]
        v = vwap(data)
        adv60 = adv(volume, 60)

        base = rank(v - ts_min(v, 12))
        exp = ts_rank(correlation(ts_rank(v, 20), ts_rank(adv60, 4), 18), 3)
        return -1.0 * (base ** exp)
