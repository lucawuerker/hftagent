"""Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

The open blend ``open*0.634196 + open*(1-0.634196)`` simplifies to just
``open``.  Fractional windows rounded to integers: delta 1, decay 15;
corr 17, decay 7, ts_rank 13.  Requires ``data["sector"]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    correlation,
    decay_linear,
    delta,
    rank,
    ts_rank,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha082(BaseFactor):
    factor_id = "alpha_082"
    name = "Alpha#82"
    category = "statistical_arbitrage"
    description = (
        "Negative min of ranked decay-linear 1-day open change and the "
        "ts-ranked decay-smoothed correlation of sector-neutralized "
        "volume with open.  Requires data['sector']."
    )
    window_length = 40
    inputs = ["open", "volume", "sector"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        volume = data["volume"]

        a = rank(decay_linear(delta(open_, 1), 15))

        vol_neut = neutralize(volume, data, "sector", "Alpha#82")
        b = ts_rank(decay_linear(correlation(vol_neut, open_, 17), 7), 13)
        return -1.0 * np.minimum(a, b)
