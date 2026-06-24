"""When price pushes outside the prior bar's range and then quickly re-enters and holds inside that range, it often indicates the breakout failed because the move lacked follow-through and liquidity was absorbed. This is analogous to the paper's safety-gain perspective: the market state that can remain inside a stable constraint region is more valuable than one that briefly escapes and gets absorbed back. I expect this to work best as a short-horizon reversal signal after failed expansion, especially when the close settles back near the open rather than near the extreme."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import delta, sign, ts_rank


@register_factor
class RangeReentrySafetyGain(BaseFactor):
    """Failed range expansion with re-entry and close-location absorption signal."""

    factor_id = "range_reentry_safety_gain"
    name = "Range Re-entry Safety Gain"
    category = "other"
    description = (
        "Measures how strongly the bar breached the prior bar's range, "
        "penalized if the close re-enters and settles deep inside that range. "
        "Positive values indicate failed excursions and intrabar absorption; "
        "negative values indicate successful range expansion and continuation."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_open = open_.shift(1)

        prev_range = (prev_high - prev_low).replace(0.0, np.nan)
        cur_range = (high - low).replace(0.0, np.nan)

        above_prev_high = (high - prev_high).clip(lower=0.0)
        below_prev_low = (prev_low - low).clip(lower=0.0)
        breach = (above_prev_high + below_prev_low) / prev_range

        reentry_from_above = (high > prev_high) & (close <= prev_high)
        reentry_from_below = (low < prev_low) & (close >= prev_low)
        reentry = reentry_from_above | reentry_from_below

        inside_prev_range = (close <= prev_high) & (close >= prev_low)
        hold_inside = inside_prev_range & reentry

        # Close location relative to the current bar, centered so values near 0.5 are neutral.
        close_loc = (close - low) / cur_range
        close_loc = close_loc.fillna(0.5)
        settle_toward_center = 1.0 - (2.0 * (close_loc - 0.5).abs())
        settle_toward_center = settle_toward_center.clip(lower=0.0, upper=1.0)

        # Prefer failed breakouts that close back inside the prior range and near the bar's midpoint/open.
        open_close_displacement = (close - open_).abs() / cur_range
        open_close_displacement = open_close_displacement.fillna(0.0)
        absorption = 1.0 - open_close_displacement.clip(lower=0.0, upper=1.0)

        # If the bar re-enters and remains inside the prior range, reward the breach more heavily.
        reentry_bonus = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        reentry_bonus = reentry_bonus.where(~hold_inside, 1.0)

        # Successful continuation gets a negative contribution when the close finishes outside the prior range.
        continuation = ((close > prev_high) | (close < prev_low)).astype(float)
        continuation = continuation * sign((close - prev_open).fillna(0.0))

        # Normalize by recent typical range of the current bar to reduce scale instability.
        typical_range = ts_rank(cur_range.replace(0.0, np.nan).fillna(0.0), 10)
        typical_range = typical_range.fillna(0.5)

        signal = (
            (breach.fillna(0.0) * (1.0 + reentry_bonus))
            * settle_toward_center.fillna(0.0)
            * absorption.fillna(0.0)
        )

        signal = signal - (0.75 * continuation.fillna(0.0) * (1.0 - settle_toward_center.fillna(0.0)))
        signal = signal * (0.5 + typical_range)

        # Smooth very small noisy values and preserve the expected numeric DataFrame shape.
        signal = signal.where(np.isfinite(signal), 0.0)
        return signal.astype(float)
