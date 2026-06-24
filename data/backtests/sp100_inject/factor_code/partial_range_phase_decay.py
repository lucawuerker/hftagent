"""A bar that travels a large fraction of its own range but fails to hold the extreme often reflects a short-lived directional impulse rather than durable price discovery. Inspired by the partial-spectrum idea in fast partial Fourier transform, this factor focuses on the lowest-frequency “shape” of recent intrabar paths by comparing the persistence of the bar's direction to its total excursion. When the path becomes more oscillatory while the close remains pinned away from the extreme, that often signals exhaustion and a higher chance of short-horizon reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_mean, ts_sum, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PartialRangePhaseDecay(BaseFactor):
    """Phase-smooth range-compression signal from recent intrabar path shape."""

    factor_id = "partial_range_phase_decay"
    name = "Partial Range Phase Decay"
    category = "other"
    description = (
        "Compute a rolling low-frequency smoothness proxy from the close-to-close path using a short window, "
        "then weight it by the signed close location within the day's range and normalize by realized bar range. "
        "The factor is high when recent movement is phase-smooth and trend-consistent, and low when the path is "
        "choppy with poor directional persistence."
    )
    window_length = 12
    inputs = ["high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0.0, np.nan)
        close_pos = ((close - low) / bar_range).clip(0.0, 1.0)
        signed_location = (close_pos * 2.0) - 1.0

        ret = delta(close, 1)
        trend = ts_sum(sign(ret), 4) / 4.0
        smoothness = ts_mean(abs(ret), 4)
        excursion = ts_sum(abs(delta(close, 1)), 4)

        phase_proxy = trend / (1.0 + smoothness + excursion)
        raw = phase_proxy * signed_location / bar_range

        return raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
