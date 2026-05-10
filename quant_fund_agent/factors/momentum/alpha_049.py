"""Alpha#49: acceleration regime — similar to Alpha#46 with -0.1 threshold.

If 10-day acceleration < -0.1: long.
Otherwise: short the 1-day close change.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha049(BaseFactor):
    factor_id = "alpha_049"
    name = "Alpha#49"
    category = "momentum"
    description = (
        "Price-acceleration regime: long when the 10-day acceleration "
        "is strongly negative (< -0.1, implying recent speed-up), "
        "otherwise short the 1-day close change."
    )
    window_length = 20
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        far = delay(close, 20)
        mid = delay(close, 10)
        accel = (far - mid) / 10.0 - (mid - close) / 10.0

        d1 = delta(close, 1)
        signal = -1.0 * d1
        signal[accel < -0.1] = 1.0
        return signal
