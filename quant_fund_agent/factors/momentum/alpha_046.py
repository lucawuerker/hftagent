"""Alpha#46: acceleration-based regime switch.

accel = (close[t-20]-close[t-10])/10 - (close[t-10]-close[t])/10
If accel > 0.25: short.
If accel < 0: long.
Else: short the 1-day close change.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha046(BaseFactor):
    factor_id = "alpha_046"
    name = "Alpha#46"
    category = "momentum"
    description = (
        "Price-acceleration regime switch: short when 10-day "
        "acceleration exceeds 0.25 (strong deceleration), long "
        "when negative (accelerating), otherwise short the daily "
        "close change."
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
        signal[accel > 0.25] = -1.0
        signal[accel < 0] = 1.0
        return signal
