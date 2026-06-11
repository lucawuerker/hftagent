"""Event factor: post-earnings-announcement drift (PEAD).

Names that beat the consensus EPS estimate tend to keep drifting up over the
following weeks.  ``epsSurprise`` (= reported − estimated EPS) is stamped at the
report date and forward-filled by the data layer, so at each date this ranks the
universe by its most recent earnings surprise — positive surprises score high.
"""

from __future__ import annotations

import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class EarningsSurpriseDrift(BaseFactor):
    factor_id = "earnings_surprise_drift"
    name = "Event: Earnings Surprise Drift"
    category = "events"
    description = (
        "Cross-sectional rank of the most recent EPS surprise (reported − "
        "estimated); positive surprises drift up (PEAD).  Requires "
        "data['epsSurprise']."
    )
    window_length = 1
    inputs = ["epsSurprise"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        surprise = data.get("epsSurprise")
        if surprise is None:  # provider advertises the field but didn't deliver it
            return data["close"] * float("nan")
        return rank(surprise)
