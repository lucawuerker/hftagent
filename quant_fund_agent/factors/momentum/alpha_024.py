"""Alpha#24: long-term 100-day trend regime switch.

If the 100-day MA change relative to lagged close <= 5%: short the
distance from 100-day low.  Otherwise: short the 3-day close delta.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delay, delta, ts_mean, ts_min
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha024(BaseFactor):
    factor_id = "alpha_024"
    name = "Alpha#24"
    category = "momentum"
    description = (
        "Long-horizon regime switch: when the 100-day moving-average "
        "drift is small (<=5%% of lagged close), short the gap from "
        "100-day low; otherwise short the 3-day close delta."
    )
    window_length = 200
    inputs = ["close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        ma100 = ts_mean(close, 100)
        drift = delta(ma100, 100) / delay(close, 100).replace(0, float("nan"))
        slow_regime = drift <= 0.05
        branch_a = -1.0 * (close - ts_min(close, 100))
        branch_b = -1.0 * delta(close, 3)
        return branch_a.where(slow_regime, branch_b)
