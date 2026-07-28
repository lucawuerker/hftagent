"""Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)

Fractional windows rounded to integers: ts_max 5, corr 5, ts_rank 3.
Requires ``data["subindustry"]`` (falls back to industry, then sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, ts_max, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha090(BaseFactor):
    factor_id = "alpha_090"
    name = "Alpha#90"
    category = "mean_reversion"
    description = (
        "Negative ranked 5-day pullback from the close high raised to "
        "the ts-ranked correlation of subindustry-neutralized ADV40 "
        "with low.  Requires data['subindustry']."
    )
    window_length = 50
    inputs = ["low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        low = data["low"]
        adv40 = adv(data["volume"], 40)

        adv_neut = neutralize(adv40, data, "subindustry", "Alpha#90")

        base = rank(close - ts_max(close, 5))
        expo = ts_rank(correlation(adv_neut, low, 5), 3)
        return -1.0 * base**expo
