"""Alpha#55: -1 * correlation(rank(((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12)))), rank(volume), 6)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank, ts_max, ts_min
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha055(BaseFactor):
    factor_id = "alpha_055"
    name = "Alpha#55"
    category = "statistical_arbitrage"
    description = (
        "Negative 6-day correlation between ranked 12-day stochastic "
        "oscillator and ranked volume.  Fades stocks where the "
        "stochastic-volume relationship is strong."
    )
    window_length = 18
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        low12 = ts_min(data["low"], 12)
        high12 = ts_max(data["high"], 12)
        stoch = (close - low12) / (high12 - low12).replace(0, float("nan"))
        return -1.0 * correlation(rank(stoch), rank(data["volume"]), 6)
