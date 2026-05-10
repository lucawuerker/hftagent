"""Alpha#2: -1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, delta, log, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha002(BaseFactor):
    factor_id = "alpha_002"
    name = "Alpha#2"
    category = "statistical_arbitrage"
    description = (
        "Negative 6-day rolling correlation between ranked 2-day "
        "log-volume change and ranked intraday return."
    )
    window_length = 8
    inputs = ["open", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        volume = data["volume"]
        x = rank(delta(log(volume), 2))
        y = rank((close - open_) / open_.replace(0, float("nan")))
        return -1.0 * correlation(x, y, 6)
