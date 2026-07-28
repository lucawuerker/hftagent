"""Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))

The low blend ``low*0.967285 + low*(1-0.967285)`` simplifies to just
``low``.  Fractional windows rounded to integers: corr 7, decay 6,
ts_rank 4; delta 3, decay 10, ts_rank 15.  Requires
``data["industry"]`` (falls back to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    delta,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha089(BaseFactor):
    factor_id = "alpha_089"
    name = "Alpha#89"
    category = "statistical_arbitrage"
    description = (
        "Spread between the ts-ranked decay-smoothed low-ADV10 "
        "correlation and the ts-ranked decay-smoothed 3-day change of "
        "industry-neutralized VWAP.  Requires data['industry']."
    )
    window_length = 35
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        low = data["low"]
        v = vwap(data)
        adv10 = adv(data["volume"], 10)

        term1 = ts_rank(decay_linear(correlation(low, adv10, 7), 6), 4)

        v_neut = neutralize(v, data, "industry", "Alpha#89")
        term2 = ts_rank(decay_linear(delta(v_neut, 3), 10), 15)
        return term1 - term2
