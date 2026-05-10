"""Alpha#73: max(rank(decay_linear(delta(vwap, 5), 3)), Ts_Rank(decay_linear(-1*delta(open*0.147+low*0.853, 2)/(open*0.147+low*0.853), 3), 17)) * -1"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import decay_linear, delta, rank, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha073(BaseFactor):
    factor_id = "alpha_073"
    name = "Alpha#73"
    category = "momentum"
    description = (
        "Negative max of ranked decay-linear 5-day VWAP delta and "
        "ts-ranked decay-linear of negative 2-day return on a "
        "low-weighted open-low blend."
    )
    window_length = 22
    inputs = ["open", "close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        low = data["low"]
        v = vwap(data)

        blend = open_ * 0.147155 + low * (1.0 - 0.147155)
        blend_ret = -1.0 * delta(blend, 2) / blend.replace(0, float("nan"))

        a = rank(decay_linear(delta(v, 5), 3))
        b = ts_rank(decay_linear(blend_ret, 3), 17)
        return -1.0 * np.maximum(a, b)
