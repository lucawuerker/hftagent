"""Large opening gaps often create an immediate overreaction, but whether the move persists or fades depends on if the bar can sustain acceptance near the extreme. When price opens far from the prior close and then spends the bar closing back toward the open or the prior close, that suggests a failed displacement and short-term mean reversion should dominate. This is inspired by the Hawkes-process idea that an initial shock can excite follow-on flow, but if the subsequent path shows poor continuation, the excitation quickly decays rather than becoming a self-reinforcing trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class GapFillPersistence(BaseFactor):
    """Gap-fill persistence / failed-displacement mean-reversion signal."""

    factor_id = "gap_fill_persistence"
    name = "Gap Fill Persistence"
    category = "mean_reversion"
    description = (
        "Measures how much of the opening gap is filled by the close relative "
        "to the bar’s full excursion. The factor is strongest when the open is "
        "far from the previous close and the close returns inside the gap, "
        "indicating rejection of the initial move."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        gap = open_ - prev_close
        gap_abs = abs_(gap)

        bar_range = (high - low).replace(0.0, np.nan)
        close_from_open = close - open_

        fill_fraction = 1.0 - (abs_(close_from_open) / (gap_abs + 1e-12))
        fill_fraction = fill_fraction.clip(lower=-2.0, upper=2.0)

        # Treat moves that close back toward the prior close as rejected gaps.
        rejection = ((close - prev_close) * gap < 0.0).astype(float)
        inside_gap = ((close >= np.minimum(open_, prev_close)) & (close <= np.maximum(open_, prev_close))).astype(float)

        # Normalize by intrabar excursion to prefer bars that retrace the gap without trend continuation.
        excursion_fill = (gap_abs / bar_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        raw = fill_fraction * rejection * inside_gap * excursion_fill

        # Emphasize larger gaps relative to recent behavior.
        gap_strength = ts_mean(rank(gap_abs), 5)
        out = (-1.0 * raw) * (0.5 + gap_strength)
        return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
