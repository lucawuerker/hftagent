"""Alpha#40: (-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha040(BaseFactor):
    factor_id = "alpha_040"
    name = "Alpha#40"
    category = "volatility"
    description = (
        "Negative ranked 10-day high volatility multiplied by the "
        "10-day high-volume correlation.  Short stocks with high "
        "volatility in highs when that volatility co-moves with volume."
    )
    window_length = 10
    inputs = ["high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        volume = data["volume"]
        return -1.0 * rank(stddev(high, 10)) * correlation(high, volume, 10)
