"""Alpha#22: -1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, rank, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha022(BaseFactor):
    factor_id = "alpha_022"
    name = "Alpha#22"
    category = "statistical_arbitrage"
    description = (
        "Negative product of the 5-day change in high-volume "
        "correlation and ranked 20-day close volatility.  Captures "
        "shifts in the high-volume relationship weighted by volatility."
    )
    window_length = 20
    inputs = ["close", "high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        hv_corr = correlation(data["high"], data["volume"], 5)
        return -1.0 * delta(hv_corr, 5) * rank(stddev(data["close"], 20))
