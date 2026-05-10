"""Alpha#13: -1 * rank(covariance(rank(close), rank(volume), 5))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import covariance, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha013(BaseFactor):
    factor_id = "alpha_013"
    name = "Alpha#13"
    category = "statistical_arbitrage"
    description = (
        "Negative ranked 5-day covariance of cross-sectionally ranked "
        "close and ranked volume.  Exploits breakdown in the "
        "price-volume co-movement structure."
    )
    window_length = 5
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * rank(covariance(rank(data["close"]), rank(data["volume"]), 5))
