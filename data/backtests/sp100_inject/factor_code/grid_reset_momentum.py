"""The dynamic grid trading paper argues that profitable participation comes from continuously resetting the reference price after a break, rather than treating the break as an exit signal. In liquid markets, a bar that closes near its high after already traversing a wide intrabar range often indicates that price discovery is still unfolding and that traders who sold into the move may be forced to chase it higher. This factor tries to capture that “reset and continue” behavior by favoring names with strong intrabar continuation after a volatility expansion, while avoiding weak closes that are more consistent with mean reversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class GridResetMomentum(BaseFactor):
    """Grid reset continuation signal from intrabar strength and range expansion."""

    factor_id = "grid_reset_momentum"
    name = "Grid Reset Momentum"
    category = "momentum"
    description = (
        "Measure the distance from open to close, scaled by the full bar range, "
        "and multiply it by an expansion term based on recent range relative to its "
        "own short-term average. The signal is positive when the bar resolves strongly "
        "in the direction of the move after a range expansion, and negative when range "
        "expansion fails to produce directional follow-through."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        directional_resolve = (close - open_) / bar_range
        directional_resolve = directional_resolve.clip(lower=-1.0, upper=1.0)

        recent_range = ts_mean(bar_range, 5)
        baseline_range = ts_mean(bar_range, 20)
        expansion = (recent_range / baseline_range.replace(0.0, np.nan)) - 1.0
        expansion = expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        signal = directional_resolve * (1.0 + expansion)
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
