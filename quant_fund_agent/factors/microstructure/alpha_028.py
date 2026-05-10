"""Alpha#28: scale((correlation(adv20, low, 5) + ((high + low) / 2)) - close)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, scale
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha028(BaseFactor):
    factor_id = "alpha_028"
    name = "Alpha#28"
    category = "microstructure"
    description = (
        "Row-normalised composite of ADV20-low correlation, midprice, "
        "and close.  Captures deviations of the close from the "
        "liquidity-adjusted midprice."
    )
    window_length = 20
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        adv20 = adv(data["volume"], 20)
        mid = (data["high"] + data["low"]) / 2.0
        raw = correlation(adv20, data["low"], 5) + mid - data["close"]
        return scale(raw)
