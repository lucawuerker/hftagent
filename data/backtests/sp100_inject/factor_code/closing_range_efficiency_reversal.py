"""When a bar travels a large intrabar range but finishes very near the open, it often reflects failed directional initiative: liquidity was consumed, but one side could not sustain price beyond the starting point. That kind of “high effort, low progress” structure tends to unwind over the next few bars as aggressive traders fade and passive liquidity refills. This is analogous to the paper’s emphasis on forecasting from sequential state evolution: the bar’s path information matters more than the endpoint alone, and inefficient paths are typically less persistent than clean trends."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, returns, ts_mean, ts_sum
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ClosingRangeEfficiencyReversal(BaseFactor):
    """High-range, low-progress bars signal short-term mean reversion."""

    factor_id = "closing_range_efficiency_reversal"
    name = "Closing Range Efficiency Reversal"
    category = "mean_reversion"
    description = (
        "Measures the mismatch between intrabar range and net close-to-open progress, "
        "normalized cross-sectionally. A positive score flags names with large ranges "
        "but weak close-through-open follow-through, which are candidates for short-term reversal."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Core intrabar path measures.
        bar_range = (high - low).abs()
        open_to_close = close - open_
        close_pos_in_range = (close - low) / (bar_range.replace(0, np.nan))
        open_pos_in_range = (open_ - low) / (bar_range.replace(0, np.nan))

        # Effort: larger ranges and higher activity indicate stronger intrabar participation.
        range_rank = rank(bar_range.fillna(0.0))
        volume_rank = rank(ts_mean(volume.fillna(0.0), 5))
        effort = 0.6 * range_rank + 0.4 * volume_rank

        # Inefficiency: large range but little close-through-open progress, especially when
        # the close sits near the opening location inside the bar range.
        progress = open_to_close / (bar_range.replace(0, np.nan))
        inefficiency = (1.0 - progress.abs().clip(lower=0.0, upper=1.0))
        location_miss = (close_pos_in_range - open_pos_in_range).abs()

        # Reversal pressure is stronger when the bar expends effort but fails to move cleanly.
        raw = (effort * (0.7 * inefficiency + 0.3 * location_miss)).fillna(0.0)

        # Optional short-term confirmation: persistent failed initiative over recent bars.
        recent_fail = ts_sum((bar_range > 0).astype(float) * (open_to_close.abs() < bar_range * 0.25).astype(float), 5)
        recent_fail = recent_fail.fillna(0.0)

        signal = raw + 0.2 * rank(recent_fail)

        # Cross-sectional normalization and sign so positive = stronger reversal candidate.
        signal = rank(signal)
        return signal.where(close.notna(), 0.0)
