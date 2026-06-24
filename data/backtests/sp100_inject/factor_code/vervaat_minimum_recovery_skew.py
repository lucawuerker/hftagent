"""The Vervaat transform shows that a path’s structure is informative around its global minimum: after the minimum, the remaining path often behaves differently from the pre-minimum segment. In markets, a bar that spends much of its range near the low but still closes above the midpoint can indicate a "re-rooting" of price away from sell-side exhaustion, while a bar that spends little time near the low but closes weakly suggests the minimum is not yet fully absorbed. This factor exploits that asymmetry by comparing recovery from the intrabar low to the location of the close within the bar, capturing a subtle continuation-vs-reversal distinction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VervaatMinimumRecoverySkew(BaseFactor):
    """Normalized recovery skew from intrabar low to close, penalized by weak open location."""

    factor_id = "vervaat_minimum_recovery_skew"
    name = "Vervaat Minimum Recovery Skew"
    category = "other"
    description = (
        "Compute a normalized recovery skew from the bar low to the close, "
        "weighted by where the close sits inside the full range and penalized "
        "when the open is already near the low. Higher values indicate stronger "
        "recovery away from the intrabar minimum, while lower values indicate "
        "weak or incomplete rebound."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0.0, np.nan)
        close_loc = (close - low) / bar_range
        open_loc = (open_ - low) / bar_range

        recovery = (close - low) / bar_range
        midpoint_bias = 2.0 * close_loc - 1.0
        open_penalty = 1.0 - open_loc

        skew = recovery * midpoint_bias * open_penalty
        skew = skew.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return skew
