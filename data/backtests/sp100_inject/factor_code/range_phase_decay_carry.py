"""When a bar closes consistently near one edge of its high-low range and the intrabar path is smooth, that move often reflects sustained order-flow imbalance rather than transient noise. The idea is to exploit the tendency for such directional pressure to persist over a short horizon after accounting for how fully the move was already “paid for” within the bar. This is inspired by the progressive-learning paper’s emphasis on separating low-frequency trend from higher-frequency variation: the factor isolates the slow component of price drift after normalizing for local volatility and path complexity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, rank, sign, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangePhaseDecayCarry(BaseFactor):
    """Range-phase decay carry from close location, range expansion, and smoothness."""

    factor_id = "range_phase_decay_carry"
    name = "Range Phase Decay Carry"
    category = "other"
    description = (
        "Cross-sectional score from close location within the bar range, "
        "adjusted by normalized range expansion and the sign of the close-to-open return."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0.0, np.nan)
        close_pos = ((close - low) / bar_range).clip(0.0, 1.0).fillna(0.5)
        open_pos = ((open_ - low) / bar_range).clip(0.0, 1.0).fillna(0.5)

        range_mid = (high + low) * 0.5
        range_centered = (close - range_mid).div(bar_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        direction = sign(close - open_).fillna(0.0)

        body = abs_(close - open_)
        smooth_path = 1.0 - ((high - np.maximum(open_, close)) + (np.minimum(open_, close) - low)).div(bar_range * 2.0)
        smooth_path = smooth_path.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

        range_expansion = ts_mean(bar_range, 5).div(ts_mean(bar_range, 20).replace(0.0, np.nan))
        range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        paid_for = body.div(bar_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        carry_raw = (close_pos - 0.5) * 2.0
        edge_bias = (close_pos - open_pos) * 2.0

        orderly = smooth_path * (1.0 - paid_for).clip(0.0, 1.0)
        decay = (2.0 - range_expansion).clip(0.0, 2.0)

        signal = (carry_raw * 0.55 + edge_bias * 0.25 + range_centered * 0.20) * direction
        signal = signal * orderly * decay
        signal = signal * (1.0 - paid_for * 0.5).clip(0.0, 1.0)

        return rank(signal.fillna(0.0))
