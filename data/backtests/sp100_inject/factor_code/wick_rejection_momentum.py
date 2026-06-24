"""When price probes an intrabar extreme but closes back near the opposite side of the bar, it often signals failed liquidation or failed breakout. That failure can attract follow-through in the direction of the close, because traders who chased the excursion are forced to unwind while informed flow keeps pressing the bar settlement. The paper’s emphasis on multi-horizon forecasting and attention over time is consistent with using this kind of intrabar state transition as a short-horizon directional feature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class WickRejectionMomentum(BaseFactor):
    """Intrabar wick rejection signal with volume modulation."""

    factor_id = "wick_rejection_momentum"
    name = "Wick Rejection Momentum"
    category = "momentum"
    description = (
        "Measures whether the bar closes near its high after rejecting the low, "
        "or near its low after rejecting the high, scaled by the full range and "
        "modulated by volume. Positive values indicate bullish rejection momentum; "
        "negative values indicate bearish rejection momentum."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12
        rng = (high - low).abs().replace(0, np.nan)

        close_pos = (close - low) / (rng + eps)
        high_reject = (high - close) / (rng + eps)
        low_reject = (close - low) / (rng + eps)

        body = close - open_
        body_ratio = body.abs() / (rng + eps)
        direction = np.sign(body)

        wick_symmetry = low_reject - high_reject
        rejection_core = ((2.0 * close_pos) - 1.0) + wick_symmetry - body_ratio * direction

        vol = volume.fillna(0.0)
        vol_med = vol.rolling(20, min_periods=1).median()
        vol_scale = (vol / (vol_med + eps)).clip(lower=0.0, upper=5.0)
        vol_mod = np.log1p(vol_scale)

        signal = rejection_core * vol_mod
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
