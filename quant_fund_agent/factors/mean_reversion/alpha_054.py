"""Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import power
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha054(BaseFactor):
    factor_id = "alpha_054"
    name = "Alpha#54"
    category = "mean_reversion"
    description = (
        "Ratio of (close-low)*open^5 to (high-low)*close^5.  "
        "Nonlinear intraday range location signal weighted by the "
        "open/close price ratio to the 5th power."
    )
    window_length = 1
    inputs = ["open", "close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        high = data["high"]
        low = data["low"]
        num = -1.0 * (low - close) * power(open_, 5.0)
        den = ((low - high) * power(close, 5.0)).replace(0, float("nan"))
        return num / den
