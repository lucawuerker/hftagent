"""When a bar closes near its extreme after spending most of the range on one side, it often signals informed order flow or an urgent repositioning event rather than random noise. The effect should be strongest when the candle body is large relative to the full range, because that suggests directional conviction instead of a mere wick-driven probe; this complements the paper’s finding that models work best in volatile, one-sided phases. In calm markets the signal should fade, while in breakout conditions it should favor continuation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, delta, rank, ts_mean


@register_factor
class RangeSkewImpulse(BaseFactor):
    """Signed intrabar impulse from range position, body conviction, and range expansion."""

    factor_id = "range_skew_impulse"
    name = "Range Skew Impulse"
    category = "momentum"
    description = (
        "Compute a signed intrabar impulse as the close's position within the high-low range, "
        "weighted by the true body-to-range ratio and scaled by recent bar range expansion. "
        "Positive values indicate bullish impulse when the close is near the high with a large body; "
        "negative values indicate bearish impulse when the close is near the low with a large body."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        full_range = (high - low).abs()
        safe_range = full_range.replace(0.0, np.nan)

        close_pos = (close - low) / (safe_range + eps)
        close_pos = close_pos.clip(0.0, 1.0)
        skew = (2.0 * close_pos - 1.0).fillna(0.0)

        body = (close - open_).abs()
        body_ratio = (body / (safe_range + eps)).clip(0.0, 1.0).fillna(0.0)

        recent_range = ts_mean(full_range, 5)
        long_range = ts_mean(full_range, 20)
        expansion = (recent_range / (long_range + eps)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        expansion = (expansion - 1.0).clip(lower=-1.0, upper=2.0)

        directional_pressure = skew * body_ratio
        impulse = directional_pressure * (1.0 + expansion)

        breakout_bias = rank(delta(close, 3).fillna(0.0))
        volatility_gate = rank(recent_range.fillna(0.0))
        calm_dampen = 0.5 + 0.5 * volatility_gate

        signal = impulse * calm_dampen * (0.5 + 0.5 * breakout_bias)
        return signal.where(np.isfinite(signal), 0.0)
