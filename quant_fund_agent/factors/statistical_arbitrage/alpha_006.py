"""Alpha#6: -1 * correlation(open, volume, 10)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha006(BaseFactor):
    factor_id = "alpha_006"
    name = "Alpha#6"
    category = "statistical_arbitrage"
    description = (
        "Negative 10-day rolling correlation between raw open price "
        "and raw volume.  Exploits breakdown of the open-volume "
        "relationship."
    )
    window_length = 10
    inputs = ["open", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return -1.0 * correlation(data["open"], data["volume"], 10)
