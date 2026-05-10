"""Alpha#23: ((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha023(BaseFactor):
    factor_id = "alpha_023"
    name = "Alpha#23"
    category = "mean_reversion"
    description = (
        "When the current high exceeds the 20-day average high "
        "(breakout), fade by returning the negative 2-day high change.  "
        "Zero otherwise.  Pure breakout mean-reversion."
    )
    window_length = 20
    inputs = ["high"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        ma20 = ts_mean(high, 20)
        fade = -1.0 * delta(high, 2)
        return fade.where(high > ma20, 0.0)
