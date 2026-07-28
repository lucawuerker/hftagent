"""Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

Fractional windows rounded to integers: delta 2, decay 3; corr 13,
decay 5, ts_rank 14.  Requires ``data["industry"]`` (falls back to
sector).
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
class Alpha087(BaseFactor):
    factor_id = "alpha_087"
    name = "Alpha#87"
    category = "statistical_arbitrage"
    description = (
        "Negative max of ranked decay-linear 2-day change of a "
        "close-vwap blend and the ts-ranked decay-smoothed absolute "
        "correlation of industry-neutralized ADV81 with close.  "
        "Requires data['industry']."
    )
    window_length = 115
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        adv81 = adv(data["volume"], 81)

        blend = close * 0.369701 + v * (1.0 - 0.369701)
        a = rank(decay_linear(delta(blend, 2), 3))

        adv_neut = neutralize(adv81, data, "industry", "Alpha#87")
        b = ts_rank(decay_linear(correlation(adv_neut, close, 13).abs(), 5), 14)
        return -1.0 * np.maximum(a, b)
