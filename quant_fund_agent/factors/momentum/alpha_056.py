"""Alpha#56: 0 - 1 * (rank(sum(returns,10) / sum(sum(returns,2),3)) * rank(returns * cap))

Requires market-cap data in ``data["cap"]``.  If absent, falls back
to volume as a liquidity proxy.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns, ts_sum
from quant_fund_agent.factors.registry import register_factor

_log = logging.getLogger(__name__)


@register_factor
class Alpha056(BaseFactor):
    factor_id = "alpha_056"
    name = "Alpha#56"
    category = "momentum"
    description = (
        "Negative product of ranked smoothed-return ratio and ranked "
        "cap-weighted returns.  Requires data['cap'] (market cap); "
        "falls back to volume if absent."
    )
    window_length = 10
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        ret = returns(data)

        inner_sum = ts_sum(ts_sum(ret, 2), 3).replace(0, float("nan"))
        ratio = ts_sum(ret, 10) / inner_sum

        if "cap" in data:
            cap = data["cap"]
        else:
            _log.warning(
                "Alpha#56: data['cap'] not provided — using volume as proxy."
            )
            cap = data["volume"]

        return -1.0 * rank(ratio) * rank(ret * cap)
