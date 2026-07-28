"""Quality factor: cross-sectional return on equity.

High-ROE (more profitable) names score high — the canonical "quality" tilt.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class QualityROE(BaseFactor):
    factor_id = "quality_roe"
    name = "Quality: Return on Equity"
    category = "fundamental"
    description = (
        "Cross-sectional rank of return on equity; more profitable names rank "
        "high.  Requires data['roe']."
    )
    window_length = 1
    inputs = ["roe", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        roe = data.get("roe")
        if roe is None:  # provider advertises the field but didn't deliver it
            return data["close"] * float("nan")
        return rank(roe)
