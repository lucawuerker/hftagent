"""When a bar closes near one extreme but spends much of its intrabar path extending the opposite way, it often reflects temporary liquidity taking the price beyond fair value before execution pressure fades. Markets tend to overreact to these asymmetric path shapes, especially when the candle body is small relative to the full range, creating a short-horizon reversal opportunity. This is consistent with the paper’s emphasis on sparse, actionable state changes: the true information is in the specific segment of the path that was “manipulated,” not the entire noisy bar path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, rank, ts_mean, ts_rank


@register_factor
class BodyShadowPressureDivergence(BaseFactor):
    """Signed divergence between close location and body/range pressure, volume-weighted."""

    factor_id = "body_shadow_pressure_divergence"
    name = "Body-Shadow Pressure Divergence"
    category = "microstructure"
    description = (
        "Compute the signed divergence between close location within the range and the "
        "intrabar body-to-range ratio, with stronger signals when volume is elevated. "
        "Positive values indicate a bar that looks bullish on the close but had weak body "
        "confirmation; negative values indicate the opposite."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        eps = 1e-12
        bar_range = (high - low).replace(0.0, np.nan)
        body = (close - open_).abs()

        close_loc = ((close - low) / (bar_range + eps)).clip(0.0, 1.0)
        body_ratio = (body / (bar_range + eps)).clip(0.0, 1.0)

        # Divergence is strongest when close is near one edge but the candle body is weak.
        signed_location = (close_loc - 0.5) * 2.0
        pressure_gap = close_loc - body_ratio
        raw_signal = signed_location * pressure_gap

        # Prefer action on bars where the candle body is small relative to the range.
        small_body = (1.0 - body_ratio).clip(0.0, 1.0)

        # Volume confirmation: compare current volume to its short-term average.
        vol_avg = adv(volume.fillna(0.0), 20)
        vol_rel = (volume / (vol_avg.replace(0.0, np.nan) + eps)).clip(lower=0.0)
        vol_strength = rank(vol_rel.fillna(0.0))
        vol_strength = vol_strength.fillna(0.0)

        # Smooth the divergence slightly to reduce bar-to-bar noise while keeping responsiveness.
        smoothed_gap = ts_mean(raw_signal.fillna(0.0), 3)
        persistence = ts_rank(small_body.fillna(0.0), 5)

        signal = smoothed_gap * (0.5 + 0.5 * vol_strength) * (0.5 + 0.5 * persistence)
        signal = signal * (0.5 + 0.5 * small_body.fillna(0.0))

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
