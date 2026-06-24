"""Measures the interaction between the overnight gap (open vs prior close) and the same-day candle body recovery toward the prior close. Long signals arise when a large gap is followed by a strong intraday reversal back toward the previous close; short signals arise for gap-downs that fail to recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OvernightBodyResetReversal(BaseFactor):
    """Fade large overnight gaps when the intraday body resets toward the prior close."""

    factor_id = "overnight_body_reset_reversal"
    name = "Overnight Body Reset Reversal"
    category = "mean_reversion"
    description = (
        "Measures the interaction between the overnight gap and same-day body recovery. "
        "Longs favor large gap-ups that fade back toward the prior close; shorts favor "
        "large gap-downs that fail to recover intraday."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        prev_close_safe = prev_close.replace(0.0, np.nan)

        overnight_gap = (open_ - prev_close) / prev_close_safe
        gap_abs = overnight_gap.abs()

        body = close - open_
        body_range = (high - low).replace(0.0, np.nan)
        body_pos = body / body_range

        recovery_to_prev_close = (close - prev_close) / prev_close_safe
        intraday_acceptance = 1.0 - ((close - low) / body_range)

        gap_rank = ts_rank(gap_abs, 20)
        recovery_rank = ts_rank((-1.0 * recovery_to_prev_close.abs()) + recovery_to_prev_close, 20)
        body_rank = ts_rank(body_pos, 20)
        range_rank = ts_rank(body_range.fillna(0.0), 20)

        raw = (gap_rank * 0.45) + (recovery_rank * 0.35) + (body_rank * 0.15) + (range_rank * 0.05)
        direction = np.sign(-overnight_gap) * np.sign(recovery_to_prev_close)
        acceptance_adjustment = ts_mean(intraday_acceptance.fillna(0.0), 5)

        signal = rank(raw) * direction * (0.5 + acceptance_adjustment.fillna(0.0))
        signal = signal.where(gap_abs.notna(), 0.0)
        return signal.astype(float)
