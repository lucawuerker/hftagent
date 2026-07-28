"""Value factor: cross-sectional earnings yield (1 / P/E).

Cheap, profitable names (high earnings yield) score high.  Negative-P/E names
(loss-makers) have no meaningful earnings yield and are excluded (NaN → dropped
from the cross-sectional rank).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ValueEarningsYield(BaseFactor):
    factor_id = "value_earnings_yield"
    name = "Value: Earnings Yield"
    category = "fundamental"
    description = (
        "Cross-sectional rank of earnings yield (1/peRatio); cheap profitable "
        "names rank high.  Requires data['peRatio']."
    )
    window_length = 1
    inputs = ["peRatio", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        pe = data.get("peRatio")
        if pe is None:  # provider advertises the field but didn't deliver it
            return data["close"] * float("nan")
        earnings_yield = 1.0 / pe.where(pe > 0)  # positive P/E only
        return rank(earnings_yield)
