"""Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha012(BaseFactor):
    factor_id = "alpha_012"
    name = "Alpha#12"
    category = "momentum"
    description = (
        "Contrarian signal: fades the 1-day close change when volume "
        "rises, follows it when volume drops.  Volume-confirmed "
        "short-term reversal / continuation filter."
    )
    window_length = 2
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return sign(delta(data["volume"], 1)) * (-1.0 * delta(data["close"], 1))
