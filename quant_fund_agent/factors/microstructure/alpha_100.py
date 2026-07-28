"""Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))

The Close Location Value (CLV) money-flow term is neutralized twice
against the same subindustry grouping, exactly as in the paper.
Requires ``data["subindustry"]`` (falls back to industry, then sector).
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors._labels import neutralize
from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, correlation, rank, scale, ts_argmin
from quant_fund_agent.factors.registry import register_factor


@register_factor
class Alpha100(BaseFactor):
    factor_id = "alpha_100"
    name = "Alpha#100"
    category = "microstructure"
    description = (
        "Negative volume-scaled combination of doubly "
        "subindustry-neutralized CLV money flow and a neutralized "
        "close-ADV20 correlation minus ranked argmin term.  Requires "
        "data['subindustry']."
    )
    window_length = 35
    inputs = ["high", "low", "close", "volume", "industry"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20).replace(0, float("nan"))

        hl_range = (high - low).replace(0, float("nan"))
        clv = ((close - low) - (high - close)) / hl_range

        flow = rank(clv * volume)
        flow_neut = neutralize(
            neutralize(flow, data, "subindustry", "Alpha#100"),
            data,
            "subindustry",
            "Alpha#100",
        )
        term1 = 1.5 * scale(flow_neut)

        raw = correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))
        term2 = scale(neutralize(raw, data, "subindustry", "Alpha#100"))

        return -1.0 * (term1 - term2) * (volume / adv20)
