"""Alpha#59: -1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, industry), volume, 4), 16), 8)

The original formula blends vwap*0.728 + vwap*(1-0.728) which simplifies
to just vwap.  Fractional windows rounded to integers.  Requires
``data["industry"]``; skips neutralisation with a warning if absent.
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
class Alpha059(BaseFactor):
    factor_id = "alpha_059"
    name = "Alpha#59"
    category = "statistical_arbitrage"
    description = (
        "Negative ts-rank of decay-linear-smoothed correlation "
        "between industry-neutralized VWAP and volume.  Variant "
        "of Alpha#58 with industry-level neutralisation and "
        "longer smoothing.  Requires data['industry']."
    )
    window_length = 40
    inputs = ["close", "high", "low", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        v = vwap(data)
        volume = data["volume"]

        if "industry" in data:
            v_neutral = indneutralize(v, data["industry"])
        else:
            warnings.warn(
                "Alpha#59: data['industry'] not provided — skipping neutralisation."
            )
            v_neutral = v

        corr = correlation(v_neutral, volume, 4)
        return -1.0 * ts_rank(decay_linear(corr, 16), 8)
