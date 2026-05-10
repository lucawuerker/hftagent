"""Alpha#25: rank((((-1 * returns) * adv20) * vwap) * (high - close))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, returns, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha025(BaseFactor):
    factor_id = "alpha_025"
    name = "Alpha#25"
    category = "momentum"
    description = (
        "Cross-sectional rank of the product of negative returns, "
        "20-day average volume, VWAP, and the high-close spread.  "
        "Captures momentum weighted by liquidity and intraday range."
    )
    window_length = 20
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ret = returns(data)
        adv20 = adv(data["volume"], 20)
        v = vwap(data)
        spread = data["high"] - data["close"]
        return rank((-1.0 * ret) * adv20 * v * spread)
