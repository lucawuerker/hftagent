"""Alpha#58: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, sector), volume, 4), 8), 6)

Fractional windows from the original paper are rounded to integers.
Requires ``data["sector"]`` (ticker → GICS sector); skips
neutralisation with a warning if absent.
"""

from __future__ import annotations

import warnings

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    correlation,
    decay_linear,
    indneutralize,
    ts_rank,
    vwap,
)
from quant_fund_agent.factors.registry import register_factor



@register_factor
class Alpha058(BaseFactor):
    factor_id = "alpha_058"
    name = "Alpha#58"
    category = "statistical_arbitrage"
    description = (
        "Negative ts-rank of decay-linear-smoothed correlation "
        "between sector-neutralized VWAP and volume.  Requires "
        "data['sector']."
    )
    window_length = 30
    inputs = ["close", "high", "low", "volume", "sector"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        volume = data["volume"]

        if "sector" in data:
            v_neutral = indneutralize(v, data["sector"])
        else:
            warnings.warn(
                "Alpha#58: data['sector'] not provided — skipping neutralisation."
            )
            v_neutral = v

        corr = correlation(v_neutral, volume, 4)
        return -1.0 * ts_rank(decay_linear(corr, 8), 6)
