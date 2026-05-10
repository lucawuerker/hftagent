"""Alpha#66: (rank(decay_linear(delta(vwap, 4), 7)) + Ts_Rank(decay_linear((low - vwap) / (open - (high+low)/2), 11), 7)) * -1

Note: the original blends low*0.967 + low*0.033 which simplifies to low.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import decay_linear, delta, rank, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha066(BaseFactor):
    factor_id = "alpha_066"
    name = "Alpha#66"
    category = "mean_reversion"
    description = (
        "Negative sum of ranked decay-linear 4-day VWAP delta and "
        "ts-ranked decay-linear of low-VWAP deviation normalised by "
        "open-midprice spread.  Captures VWAP trend and low-side "
        "pressure."
    )
    window_length = 18
    inputs = ["open", "close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        v = vwap(data)

        mid = (high + low) / 2.0
        denom = (open_ - mid).replace(0, float("nan"))

        a = rank(decay_linear(delta(v, 4), 7))
        b = ts_rank(decay_linear((low - v) / denom, 11), 7)
        return -1.0 * (a + b)
