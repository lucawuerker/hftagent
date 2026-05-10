"""Alpha#10: cross-sectionally ranked version of Alpha#9 with 4-day horizon."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, ts_max, ts_min
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha010(BaseFactor):
    factor_id = "alpha_010"
    name = "Alpha#10"
    category = "momentum"
    description = (
        "Cross-sectional rank of a regime-conditioned 1-day close delta "
        "using a 4-day min/max filter.  Ranked variant of Alpha#9 with "
        "a shorter lookback."
    )
    window_length = 5
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        d1 = delta(data["close"], 1)
        trending = (0 < ts_min(d1, 4)) | (ts_max(d1, 4) < 0)
        return rank(d1.where(trending, -1.0 * d1))
