"""Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

Fractional windows rounded to integers: delta 1, corr 18, ts_rank 18.
Requires ``data["industry"]`` (falls back to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha070(BaseFactor):
    factor_id = "alpha_070"
    name = "Alpha#70"
    category = "momentum"
    description = (
        "Negative ranked 1-day VWAP change raised to the ts-ranked "
        "correlation of industry-neutralized close with ADV50.  "
        "Requires data['industry']."
    )
    window_length = 90
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        adv50 = adv(data["volume"], 50)

        close_neut = neutralize(close, data, "industry", "Alpha#70")

        base = rank(delta(v, 1))
        expo = ts_rank(correlation(close_neut, adv50, 18), 18)
        return -1.0 * base**expo
