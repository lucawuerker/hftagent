"""Alpha#7: (adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7))) : -1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, adv, delta, sign, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha007(BaseFactor):
    factor_id = "alpha_007"
    name = "Alpha#7"
    category = "momentum"
    description = (
        "On high-volume days (volume > 20-day ADV), returns the signed "
        "60-day time-series rank of the 7-day close delta magnitude.  "
        "Momentum signal conditioned on unusual activity."
    )
    window_length = 60
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        d7 = delta(close, 7)
        active = (-1.0 * ts_rank(abs_(d7), 60)) * sign(d7)
        fallback = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
        return active.where(adv20 < volume, fallback)
