"""Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)

Fractional windows rounded to integers: delta 3, ts_max 5, corr 5,
ts_rank 9.  Requires ``data["industry"]`` (falls back to sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, delta, rank, ts_max, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha069(BaseFactor):
    factor_id = "alpha_069"
    name = "Alpha#69"
    category = "momentum"
    description = (
        "Negative ranked 5-day max of industry-neutralized VWAP changes "
        "raised to the ts-ranked correlation of a close-vwap blend with "
        "ADV20.  Requires data['industry']."
    )
    window_length = 40
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        adv20 = adv(data["volume"], 20)

        v_neut = neutralize(v, data, "industry", "Alpha#69")
        base = rank(ts_max(delta(v_neut, 3), 5))

        blend = close * 0.490655 + v * (1.0 - 0.490655)
        expo = ts_rank(correlation(blend, adv20, 5), 9)
        return -1.0 * base**expo
