"""Alpha#48: indneutralize((correlation(delta(close,1), delta(delay(close,1),1), 250) * delta(close,1)) / close, subindustry) / sum((delta(close,1)/delay(close,1))^2, 250)

Requires a ``subindustry`` Series (ticker → GICS sub-industry label)
passed via ``data["subindustry"]``.  If absent the neutralisation step
is skipped and a warning is logged.
"""

from __future__ import annotations

import warnings

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    correlation,
    delay,
    delta,
    indneutralize,
    power,
    ts_sum,
)
from quant_fund_agent.factors.registry import register_factor



@register_factor
class Alpha048(BaseFactor):
    factor_id = "alpha_048"
    name = "Alpha#48"
    category = "statistical_arbitrage"
    description = (
        "Industry-neutralized 250-day return-autocorrelation signal "
        "scaled by close, divided by 250-day sum of squared daily "
        "returns.  Requires subindustry classification in "
        "data['subindustry']."
    )
    window_length = 252
    inputs = ["close", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        d1 = delta(close, 1)
        d1_lag = delta(delay(close, 1), 1)

        numerator = correlation(d1, d1_lag, 250) * d1 / close.replace(0, float("nan"))

        if "subindustry" in data:
            labels = data.get("subindustry")
            if labels is None or getattr(labels, "empty", False):
                labels = data.get("industry")
            numerator = indneutralize(numerator, labels)
        else:
            warnings.warn(
                "Alpha#48: data['subindustry'] not provided — "
                "skipping industry neutralisation."
            )

        ret_sq = power(d1 / delay(close, 1).replace(0, float("nan")), 2.0)
        denominator = ts_sum(ret_sq, 250).replace(0, float("nan"))
        return numerator / denominator
