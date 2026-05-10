"""Alpha#53: -1 * delta(((close - low) - (high - close)) / (close - low), 9)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha053(BaseFactor):
    factor_id = "alpha_053"
    name = "Alpha#53"
    category = "mean_reversion"
    description = (
        "Negative 9-day change in the Williams %R-style close "
        "location within the daily range.  Fades shifts in "
        "where the close sits inside the high-low range."
    )
    window_length = 10
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        hl_range = (close - low).replace(0, float("nan"))
        loc_value = ((close - low) - (high - close)) / hl_range
        return -1.0 * delta(loc_value, 9)
