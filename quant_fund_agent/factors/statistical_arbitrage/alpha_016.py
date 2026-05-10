"""Alpha#16: -1 * rank(covariance(rank(high), rank(volume), 5))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import covariance, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha016(BaseFactor):
    factor_id = "alpha_016"
    name = "Alpha#16"
    category = "statistical_arbitrage"
    description = (
        "Negative ranked 5-day covariance of cross-sectionally ranked "
        "highs and ranked volume.  Similar structure to Alpha#13 but "
        "uses high instead of close."
    )
    window_length = 5
    inputs = ["high", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * rank(covariance(rank(data["high"]), rank(data["volume"]), 5))
