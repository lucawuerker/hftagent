"""Inspired by the near-extreme density framework in Politi et al. (2011), this factor asks whether a bar finishes close to its own intrabar extreme after normalizing for how much of the range was actually explored. When the close sits very near the high after an unusually one-sided range, it often reflects strong latent order-flow imbalance and continuation potential; when it sits near the low, the same logic applies in the opposite direction. Using a normalized near-extreme measure rather than raw return helps reduce sensitivity to price level and makes the signal more robust across names with different volatilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_sum, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class NearExtremeImbalancePressure(BaseFactor):
    """Close-to-extreme pressure adjusted by range exploration and volume support."""

    factor_id = "near_extreme_imbalance_pressure"
    name = "Near-Extreme Imbalance Pressure"
    category = "microstructure"
    description = (
        "Signed close-location score within the intrabar range, scaled by range "
        "usage and penalized when volume does not support the extreme; cross-"
        "sectionally ranked each bar."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        true_range = (high - low).replace(0.0, np.nan)
        close_pos = ((close - low) / true_range).clip(lower=0.0, upper=1.0)
        signed_extreme = 2.0 * close_pos - 1.0

        # Range exploration: small when the bar spends most of its movement near one end.
        range_span = (high - low).abs()
        bar_body = (close - data["open"].astype(float)).abs()
        exploration = 1.0 - (bar_body / range_span.replace(0.0, np.nan))
        exploration = exploration.fillna(0.0).clip(lower=0.0, upper=1.0)

        # Support from trading activity: compare current volume to a short average.
        vol_avg = ts_mean(volume, 10)
        vol_ratio = (volume / vol_avg.replace(0.0, np.nan)).fillna(0.0)
        vol_support = (vol_ratio - 1.0).clip(lower=-1.0, upper=3.0)

        # A mild VWAP sanity check: if close extends beyond vwap in the direction of the extreme,
        # reward it; otherwise dampen it.
        pv = vwap(data).astype(float)
        vwap_bias = ((close - pv) / true_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vwap_alignment = np.sign(signed_extreme) * vwap_bias

        raw_score = signed_extreme * (0.5 + 0.5 * exploration) * (1.0 + 0.25 * vol_support) * (1.0 + 0.25 * vwap_alignment)
        raw_score = raw_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional normalization to make the score comparable across names.
        centered = raw_score - ts_mean(raw_score, 1)
        out = rank(centered)
        return out
