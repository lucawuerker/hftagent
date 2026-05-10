"""Alpha#27: (0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0)) ? -1 : 1"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import correlation, rank, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha027(BaseFactor):
    factor_id = "alpha_027"
    name = "Alpha#27"
    category = "microstructure"
    description = (
        "Binary signal based on the ranked 2-day average of the 6-day "
        "correlation between ranked volume and ranked VWAP.  Short "
        "when the correlation rank is high, long otherwise."
    )
    window_length = 8
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        corr = correlation(rank(data["volume"]), rank(v), 6)
        scored = rank(ts_sum(corr, 2) / 2.0)
        signal = pd.DataFrame(1.0, index=scored.index, columns=scored.columns)
        signal[scored > 0.5] = -1.0
        return signal
