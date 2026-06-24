"""Inspired by the polling-system idea that the socially optimal policy depends on where the server is in its cycle, this factor treats each bar as a micro-cycle and asks whether price is drifting toward the bar’s most informative extreme. When the close is persistently nearer the high after a strong opening-to-close advance, it suggests continuation through local order-flow dominance; when the close is nearer the low after a similar decline, it suggests the opposite. The edge should come from short-horizon behavioral underreaction to intrabar path information, which is not captured by plain close-to-close momentum or range-based reversal signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CyclePositionDriftBias(BaseFactor):
    """Close-location drift signal adjusted for signed body strength and range expansion."""

    factor_id = "cycle_position_drift_bias"
    name = "Cycle Position Drift Bias"
    category = "other"
    description = (
        "Compute a cross-sectional score from the normalized close location "
        "within the bar range, adjusted by the signed body direction and "
        "range expansion versus the recent average range. Higher values "
        "indicate closes that finish near the high with supportive range "
        "expansion; lower values indicate closes that finish near the low "
        "with supportive range expansion."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0, np.nan)
        close_location = ((close - low) / bar_range).clip(0.0, 1.0)
        close_location = close_location.fillna(0.5)

        body = close - open_
        body_range = (body / bar_range).clip(-1.0, 1.0)
        body_range = body_range.fillna(0.0)

        range_size = (high - low).fillna(0.0)
        avg_range = ts_mean(range_size, 20).replace(0, np.nan)
        range_expansion = (range_size / avg_range - 1.0).replace([np.inf, -np.inf], np.nan)
        range_expansion = range_expansion.fillna(0.0)

        signed_location = (2.0 * close_location - 1.0) * np.sign(body).replace(0, 1.0)
        signed_location = signed_location.fillna(0.0)

        body_strength = rank(body_range.abs()) * np.sign(body_range)
        body_strength = body_strength.fillna(0.0)

        expansion_support = rank(range_expansion)
        expansion_support = (2.0 * expansion_support - 1.0).fillna(0.0)

        drift_raw = signed_location * (1.0 + body_strength) * (1.0 + 0.5 * expansion_support)
        drift_raw = drift_raw * (1.0 + range_expansion.clip(lower=-0.75, upper=2.0))

        return drift_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
