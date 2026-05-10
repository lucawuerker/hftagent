"""Alpha#21: Bollinger-band regime with volume override.

If 2-day MA > upper band (8-day MA + std): short.
If 2-day MA < lower band (8-day MA - std): long.
Otherwise, if relative volume >= 1: long, else short.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, stddev, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha021(BaseFactor):
    factor_id = "alpha_021"
    name = "Alpha#21"
    category = "volatility"
    description = (
        "Bollinger-band regime switch: short when 2-day close mean "
        "exceeds the 8-day upper band, long when below the lower band, "
        "else follow relative volume direction."
    )
    window_length = 20
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        ma8 = ts_mean(close, 8)
        std8 = stddev(close, 8)
        ma2 = ts_mean(close, 2)
        adv20 = adv(volume, 20)

        upper = ma8 + std8
        lower = ma8 - std8
        rel_vol = volume / adv20.replace(0, float("nan"))

        signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        signal[ma2 > upper] = -1.0
        signal[ma2 < lower] = 1.0
        neutral = (ma2 <= upper) & (ma2 >= lower)
        signal[neutral & (rel_vol >= 1.0)] = 1.0
        signal[neutral & (rel_vol < 1.0)] = -1.0
        return signal
