"""Alpha#60: 0 - 1 * (2*scale(rank(((close-low)-(high-close))/(high-low)*volume)) - scale(rank(ts_argmax(close,10))))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, scale, ts_argmax
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha060(BaseFactor):
    factor_id = "alpha_060"
    name = "Alpha#60"
    category = "microstructure"
    description = (
        "Negative composite of scaled ranked Chaikin-style "
        "money-flow volume and scaled ranked 10-day close argmax.  "
        "Captures volume-weighted close-location imbalance relative "
        "to the position of the recent high."
    )
    window_length = 10
    inputs = ["close", "high", "low", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        hl_range = (high - low).replace(0, float("nan"))
        clv = ((close - low) - (high - close)) / hl_range
        mf_vol = rank(clv * volume)
        argmax_rank = rank(ts_argmax(close, 10))

        return -1.0 * (2.0 * scale(mf_vol) - scale(argmax_rank))
