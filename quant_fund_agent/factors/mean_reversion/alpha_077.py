"""Alpha#77: min(rank(decay_linear(((high+low)/2 + high) - (vwap+high), 20)), rank(decay_linear(corr((high+low)/2, adv40, 3), 6)))"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, decay_linear, rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha077(BaseFactor):
    factor_id = "alpha_077"
    name = "Alpha#77"
    category = "mean_reversion"
    description = (
        "Min of ranked decay-linear midprice-VWAP deviation "
        "(simplifies to (low/2 - vwap/2)) and ranked decay-linear "
        "midprice-ADV40 correlation.  Captures the weaker of two "
        "fair-value reversion signals."
    )
    window_length = 46
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        volume = data["volume"]
        v = vwap(data)
        adv40 = adv(volume, 40)
        mid = (high + low) / 2.0

        spread = (mid + high) - (v + high)  # simplifies to mid - v
        a = rank(decay_linear(spread, 20))
        b = rank(decay_linear(correlation(mid, adv40, 3), 6))
        return np.minimum(a, b)
