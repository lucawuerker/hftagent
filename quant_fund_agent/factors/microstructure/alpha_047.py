"""Alpha#47: ((rank(1/close)*volume/adv20) * (high*rank(high-close)/(sum(high,5)/5))) - rank(vwap - delay(vwap,5))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, delay, rank, ts_mean, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha047(BaseFactor):
    factor_id = "alpha_047"
    name = "Alpha#47"
    category = "microstructure"
    description = (
        "Composite of inverse-price-weighted relative volume times "
        "the high-close deviation normalised by 5-day average high, "
        "minus ranked 5-day VWAP change.  Microstructure signal "
        "combining liquidity, range, and fair-value momentum."
    )
    window_length = 20
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        v = vwap(data)

        inv_price = rank(1.0 / close.replace(0, float("nan")))
        rel_vol = volume / adv20.replace(0, float("nan"))
        range_part = high * rank(high - close) / ts_mean(high, 5).replace(0, float("nan"))

        lhs = inv_price * rel_vol * range_part
        rhs = rank(v - delay(v, 5))
        return lhs - rhs
