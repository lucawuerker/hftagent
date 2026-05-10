"""Alpha#18: -1 * rank((stddev(abs(close - open), 5) + (close - open)) + correlation(close, open, 10))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, correlation, rank, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha018(BaseFactor):
    factor_id = "alpha_018"
    name = "Alpha#18"
    category = "mean_reversion"
    description = (
        "Negative ranked composite of 5-day intraday-range volatility, "
        "current intraday return, and 10-day close-open correlation.  "
        "Fades stocks with elevated intraday dislocations."
    )
    window_length = 10
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        spread = close - open_
        x = stddev(abs_(spread), 5) + spread + correlation(close, open_, 10)
        return -1.0 * rank(x)
