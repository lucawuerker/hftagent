"""Alpha#51: acceleration regime with -0.05 threshold.

If 10-day acceleration < -0.05: long.
Otherwise: short the 1-day close change.
Variant of Alpha#46/#49 with a different threshold.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, delta
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha051(BaseFactor):
    factor_id = "alpha_051"
    name = "Alpha#51"
    category = "momentum"
    description = (
        "Price-acceleration regime: long when 10-day acceleration "
        "is below -0.05 (recent speed-up), otherwise short the "
        "1-day close change.  Tighter threshold variant of #49."
    )
    window_length = 20
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        far = delay(close, 20)
        mid = delay(close, 10)
        accel = (far - mid) / 10.0 - (mid - close) / 10.0

        signal = -1.0 * delta(close, 1)
        signal[accel < -0.05] = 1.0
        return signal
