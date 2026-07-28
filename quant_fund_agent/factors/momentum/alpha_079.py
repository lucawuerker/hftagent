"""Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))

Fractional windows rounded to integers: delta 1, ts_rank(vwap) 4,
ts_rank(adv150) 9, corr 15.  Boolean output is cast to {0, 1} floats
(cf. Alpha#68).  Requires ``data["sector"]``.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha079(BaseFactor):
    factor_id = "alpha_079"
    name = "Alpha#79"
    category = "momentum"
    description = (
        "Binary signal: ranked 1-day change of a sector-neutralized "
        "close-open blend compared against the ranked correlation of "
        "ts-ranked VWAP with ts-ranked ADV150.  Requires "
        "data['sector']."
    )
    window_length = 180
    inputs = ["open", "high", "low", "close", "volume", "sector"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        v = vwap(data)
        adv150 = adv(data["volume"], 150)

        blend = close * 0.60733 + open_ * (1.0 - 0.60733)
        blend_neut = neutralize(blend, data, "sector", "Alpha#79")

        lhs = rank(delta(blend_neut, 1))
        rhs = rank(correlation(ts_rank(v, 4), ts_rank(adv150, 9), 15))
        return (lhs < rhs).astype(float)
