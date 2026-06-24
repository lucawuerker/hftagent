"""Cont & de Larrard argue that short-horizon price moves are driven by the relative depletion/replenishment of the best bid and ask queues, with imbalance in order flow determining the next price move. When a bar closes near its high on strong turnover, it often reflects ask-side depletion and a higher probability that the next price impulse continues upward; conversely, a close near the low with heavy volume points to bid-side exhaustion and downside continuation. This factor tries to isolate that queue-pressure state from ordinary trend signals by combining close-location and volume intensity into a directional microstructure score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarQueuePressure(BaseFactor):
    """Directional intrabar queue-pressure signal from close location and volume intensity."""

    factor_id = "intrabar_queue_pressure"
    name = "Intrabar Queue Pressure"
    category = "microstructure"
    description = (
        "Compute a signed pressure measure from the bar's close location within the range, "
        "scaled by volume relative to its recent average and gated by range expansion. The "
        "output is cross-sectional: higher values indicate stronger inferred buy-side queue "
        "depletion / upward pressure, lower values indicate sell-side pressure."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0.0, np.nan)
        close_loc = ((close - low) / rng).clip(0.0, 1.0)
        directional = (close_loc - 0.5) * 2.0

        vol_avg = adv(volume, 20)
        vol_intensity = (volume / vol_avg.replace(0.0, np.nan)) - 1.0
        vol_intensity = vol_intensity.fillna(0.0)

        bar_range = (high - low)
        avg_range = ts_mean(bar_range, 20)
        range_expansion = (bar_range / avg_range.replace(0.0, np.nan)) - 1.0
        range_expansion = range_expansion.fillna(0.0)

        body = (close - open_).abs()
        body_frac = body / rng.replace(0.0, np.nan)
        body_frac = body_frac.fillna(0.0)

        pressure = directional * (1.0 + vol_intensity) * (1.0 + range_expansion)
        pressure = pressure * (0.5 + 0.5 * body_frac)

        cross_sectional = rank(pressure)
        return (cross_sectional * 2.0) - 1.0
