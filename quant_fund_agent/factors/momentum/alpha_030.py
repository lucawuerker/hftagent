"""Alpha#30: ((1 - rank(sign(close-delay(close,1)) + sign(delay(close,1)-delay(close,2)) + sign(delay(close,2)-delay(close,3)))) * sum(volume,5)) / sum(volume,20)"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, rank, sign, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha030(BaseFactor):
    factor_id = "alpha_030"
    name = "Alpha#30"
    category = "momentum"
    description = (
        "Contrarian 3-day consecutive-direction score (ranked and "
        "inverted), scaled by the 5-day vs 20-day volume ratio.  "
        "Fades sustained directional moves when recent volume is "
        "elevated relative to the longer window."
    )
    window_length = 20
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        d1 = sign(close - delay(close, 1))
        d2 = sign(delay(close, 1) - delay(close, 2))
        d3 = sign(delay(close, 2) - delay(close, 3))
        direction_score = 1.0 - rank(d1 + d2 + d3)
        vol_ratio = ts_sum(volume, 5) / ts_sum(volume, 20).replace(0, float("nan"))
        return direction_score * vol_ratio
