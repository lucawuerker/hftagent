"""Alpha#9: regime-conditioned delta(close, 1) using 5-day min/max filters."""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_max, ts_min
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha009(BaseFactor):
    factor_id = "alpha_009"
    name = "Alpha#9"
    category = "momentum"
    description = (
        "If the 5-day minimum of daily close changes is positive "
        "(sustained uptrend) or the 5-day maximum is negative "
        "(sustained downtrend), follow the trend; otherwise, fade it."
    )
    window_length = 6
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        d1 = delta(data["close"], 1)
        trending = (0 < ts_min(d1, 5)) | (ts_max(d1, 5) < 0)
        return d1.where(trending, -1.0 * d1)
