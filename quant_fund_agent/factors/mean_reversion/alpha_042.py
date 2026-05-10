"""Alpha#42: rank((vwap - close)) / rank((vwap + close))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha042(BaseFactor):
    factor_id = "alpha_042"
    name = "Alpha#42"
    category = "mean_reversion"
    description = (
        "Ratio of ranked VWAP-close spread to ranked VWAP+close sum.  "
        "Positive when close is cheap relative to VWAP in cross-section."
    )
    window_length = 1
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        v = vwap(data)
        num = rank(v - close)
        den = rank(v + close).replace(0, float("nan"))
        return num / den
