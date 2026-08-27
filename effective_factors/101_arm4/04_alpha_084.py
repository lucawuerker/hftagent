"""Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15)), 21), delta(close, 5))"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, signed_power, ts_max, ts_rank, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha084(BaseFactor):
    prediction_horizon = 6
    factor_id = "alpha_084"
    name = "Alpha#84"
    category = "mean_reversion"
    description = (
        "Signed power of 21-day ts-rank of VWAP distance from its "
        "15-day max, with exponent equal to the 5-day close delta.  "
        "Nonlinear mean-reversion around the VWAP ceiling."
    )
    window_length = 36
    inputs = ["close", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        base = ts_rank(v - ts_max(v, 15), 21)
        exp = delta(data["close"], 5)
        return signed_power(base, exp)
