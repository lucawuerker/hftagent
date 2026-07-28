"""Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

Fractional windows rounded to integers: delta 4, corr 5, ts_rank 6.
Requires ``data["industry"]`` (falls back to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, sign, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha080(BaseFactor):
    factor_id = "alpha_080"
    name = "Alpha#80"
    category = "momentum"
    description = (
        "Negative ranked sign of the 4-day change of an "
        "industry-neutralized open-high blend raised to the ts-ranked "
        "high-ADV10 correlation.  Requires data['industry']."
    )
    window_length = 25
    inputs = ["open", "high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        adv10 = adv(data["volume"], 10)

        blend = open_ * 0.868128 + high * (1.0 - 0.868128)
        blend_neut = neutralize(blend, data, "industry", "Alpha#80")

        base = rank(sign(delta(blend_neut, 4)))
        expo = ts_rank(correlation(high, adv10, 5), 6)
        return -1.0 * base**expo
