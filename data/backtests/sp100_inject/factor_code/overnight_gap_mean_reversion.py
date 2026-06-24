"""Measure the signed gap from previous close to current open, then downweight it when the bar closes back toward the prior close. A simple implementation is the cross-sectional z-score of −(open / close.shift(1) − 1) multiplied by a close-to-open retracement term."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import rank, delay


@register_factor
class OvernightGapMeanReversion(BaseFactor):
    """Fade large overnight gaps when the open-to-close move retraces toward the prior close."""

    factor_id = "overnight_gap_mean_reversion"
    name = "Overnight Gap Mean Reversion"
    category = "mean_reversion"
    description = (
        "Cross-sectional mean-reversion signal that fades the prior close-to-open gap, "
        "scaled by how much the bar closes back toward the prior close."
    )
    window_length = 2
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        prev_close = delay(close, 1)

        safe_prev_close = prev_close.replace(0, np.nan)
        gap = open_ / safe_prev_close - 1.0
        gap = gap.replace([np.inf, -np.inf], np.nan)

        # Retracement term is strongest when close moves back toward the prior close.
        retracement = 1.0 - (close - open_).abs() / (open_ - prev_close).abs().replace(0, np.nan)
        retracement = retracement.clip(lower=0.0, upper=1.0)
        retracement = retracement.fillna(0.0)

        signal = rank(-gap) * retracement
        signal = signal.replace([np.inf, -np.inf], np.nan)
        return signal.fillna(0.0)
