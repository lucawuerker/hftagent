"""Hawkes Gap Excitation Imbalance: cluster-weighted bullish minus bearish shock excitation from range and volume impulses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (
    abs_,
    decay_linear,
    rank,
    ts_mean,
    ts_sum,
)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class HawkesGapExcitationImbalance(BaseFactor):
    """Shock-cluster imbalance from bullish versus bearish excitation intensity."""

    factor_id = "hawkes_gap_excitation_imbalance"
    name = "Hawkes Gap Excitation Imbalance"
    category = "other"
    description = (
        "Compute a per-ticker excitation score from recent bars using exponentially decayed counts "
        "of bars where true range and volume are both elevated relative to their rolling history, "
        "then weight those events by signed candle direction (close-open) and close location within "
        "the range. The final factor is the cross-sectional normalized difference between bullish and "
        "bearish excitation intensity, producing positive values when upward shocks are more "
        "self-reinforcing than downward shocks."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        prev_close = close.shift(1)
        true_range = (high.combine(prev_close, np.maximum) - low.combine(prev_close, np.minimum)).abs()
        candle_range = (high - low).abs().replace(0, np.nan)

        range_baseline = ts_mean(true_range, 10).replace(0, np.nan)
        volume_baseline = ts_mean(volume, 10).replace(0, np.nan)
        close_baseline = ts_mean(close, 10).replace(0, np.nan)

        range_ratio = true_range / range_baseline
        volume_ratio = volume / volume_baseline
        gap_ratio = (close - open_).abs() / close_baseline

        shock_event = (
            (range_ratio > 1.25)
            & (volume_ratio > 1.10)
            & (gap_ratio > 0.002)
        ).astype(float)

        event_intensity = (range_ratio.fillna(0.0) * volume_ratio.fillna(0.0)).where(shock_event > 0.0, 0.0)
        excitation = decay_linear(event_intensity, 8).fillna(0.0)

        candle_direction = np.sign(close - open_)
        close_location = ((close - low) / candle_range).clip(0.0, 1.0).fillna(0.5)
        directional_bias = ((close_location - 0.5) * 2.0) * candle_direction.fillna(0.0)

        bullish_event = event_intensity.where(directional_bias > 0.0, 0.0)
        bearish_event = event_intensity.where(directional_bias < 0.0, 0.0)

        bullish_excitation = decay_linear(bullish_event, 8).fillna(0.0)
        bearish_excitation = decay_linear(bearish_event, 8).fillna(0.0)

        bullish_cluster = ts_sum((shock_event > 0.0).astype(float), 5).fillna(0.0)
        bearish_cluster = ts_sum((shock_event > 0.0).astype(float) * (directional_bias < 0.0).astype(float), 5).fillna(0.0)

        imbalance = bullish_excitation - bearish_excitation
        cluster_boost = (bullish_cluster - bearish_cluster) / 5.0

        raw = (imbalance * (1.0 + cluster_boost)).fillna(0.0)
        normalized = rank(raw)
        centered = (normalized - 0.5) * 2.0

        excitation_scale = ts_mean(excitation, 5).fillna(0.0)
        out = centered * (1.0 + excitation_scale)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
