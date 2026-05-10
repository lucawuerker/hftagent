"""Alpha#41: ((high * low)^0.5) - vwap"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha041(BaseFactor):
    factor_id = "alpha_041"
    name = "Alpha#41"
    category = "mean_reversion"
    description = (
        "Geometric midprice (sqrt(high*low)) minus VWAP.  Captures "
        "deviation between the range-based fair value and the "
        "volume-weighted fair value."
    )
    window_length = 1
    inputs = ["high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        geo_mid = np.sqrt(data["high"] * data["low"])
        return geo_mid - vwap(data)
