"""Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)

Fractional windows rounded to integers: delta 1, decay 12; corr 8,
ts_rank 20, decay 17, ts_rank 19.  Requires ``data["sector"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    adv,
    correlation,
    decay_linear,
    delta,
    rank,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha076(BaseFactor):
    factor_id = "alpha_076"
    name = "Alpha#76"
    category = "statistical_arbitrage"
    description = (
        "Negative max of ranked decay-linear 1-day VWAP change and a "
        "doubly ts-ranked, decay-smoothed correlation of "
        "sector-neutralized low with ADV81.  Requires data['sector']."
    )
    window_length = 150
    inputs = ["high", "low", "close", "volume", "sector"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        low = data["low"]
        v = vwap(data)
        adv81 = adv(data["volume"], 81)

        a = rank(decay_linear(delta(v, 1), 12))

        low_neut = neutralize(low, data, "sector", "Alpha#76")
        b = ts_rank(
            decay_linear(ts_rank(correlation(low_neut, adv81, 8), 20), 17), 19
        )
        return -1.0 * np.maximum(a, b)
