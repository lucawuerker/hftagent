"""Prices that advance through the bar with participation across both the bar body and the extremes tend to indicate broad, multi-horizon agreement rather than a fragile one-leg move. Inspired by the paper’s multi-resolution / wavelet framing, this factor looks for cross-scale persistence in intrabar direction: when close-to-open direction is aligned with both excursion depth and end-of-bar location, continuation is more likely than a noisy reversal. The edge should be strongest when the move is not just large, but structurally broad across the full OHLC range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarWaveletBreadthMomentum(BaseFactor):
    """Cross-sectional intrabar breadth momentum from OHLC alignment."""

    factor_id = "intrabar_wavelet_breadth_momentum"
    name = "Intrabar Wavelet Breadth Momentum"
    category = "momentum"
    description = (
        "Compute a cross-sectional score from the alignment of intrabar return "
        "direction with range breadth: signed close-open return multiplied by a "
        "normalized body-to-range ratio and a normalized close-location term. "
        "Higher values indicate a trend that is consistent across multiple "
        "intrabar scales of the bar."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        rng = (high - low).replace(0.0, np.nan)
        body = close - open_

        # Signed intrabar direction scaled by body participation in the full range.
        body_strength = body / rng

        # End-of-bar location within the intrabar range, centered to [-1, 1].
        close_location = ((close - low) / rng) * 2.0 - 1.0

        # Excursion breadth: larger bodies that also consume more of the bar range.
        breadth = (body.abs() / rng)

        raw = body_strength * close_location * breadth

        # Keep the signal numerically stable and bounded for cross-sectional use.
        signal = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal.clip(lower=-10.0, upper=10.0)
