"""Bars that travel far away from the open but fail to hold the extreme into the close often reflect temporary liquidity demand or stop-driven price pressure rather than durable information. A large upper wick after an intrabar rally, or a large lower wick after an intrabar selloff, suggests exhaustion and a higher chance of next-bar mean reversion as inventory providers fade the move. This complements the seed ensemble by focusing on intrabar rejection rather than simple close-location effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeWickReversal(BaseFactor):
    """Signed asymmetry of bar wicks relative to total range."""

    factor_id = "range_wick_reversal"
    name = "Range Wick Reversal"
    category = "mean_reversion"
    description = (
        "Compute the signed asymmetry of the bar's wicks relative to its total range, "
        "emphasizing rejection at the extremes. Positive values indicate bearish "
        "rejection of the high; negative values indicate bullish rejection of the low."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0.0, np.nan)
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)

        wick_asymmetry = (upper_wick - lower_wick) / bar_range
        body_position = ((close - open_) / bar_range).clip(lower=-1.0, upper=1.0)

        signal = wick_asymmetry + 0.25 * body_position
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
