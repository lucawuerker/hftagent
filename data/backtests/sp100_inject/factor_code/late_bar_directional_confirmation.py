"""A bar that spends most of its intrabar path in one direction and finishes near the directional extreme tends to reflect informed pressure rather than random noise. When the close is reinforced by a strong late-bar push, continuation is more likely because liquidity was not sufficient to fully absorb the move before settlement. This is conceptually aligned with CLAM’s emphasis on latent continuous actions: the path taken matters, not just the endpoint, so a signal that rewards orderly, persistent intrabar motion should capture short-horizon follow-through."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LateBarDirectionalConfirmation(BaseFactor):
    """Cross-sectional rank of intrabar directional confirmation weighted by relative range."""

    factor_id = "late_bar_directional_confirmation"
    name = "Late-Bar Directional Confirmation"
    category = "momentum"
    description = (
        "Measure the signed intrabar trend as a combination of close-to-open return "
        "and where the close sits within the bar’s high-low range, then weight it by "
        "bar range relative to recent typical range to emphasize decisive bars. "
        "Cross-sectionally rank the result each bar."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        intrabar_range = (high - low).replace(0, np.nan)
        close_to_open = (close - open_) / open_.replace(0, np.nan)
        close_location = ((close - low) / intrabar_range).clip(0.0, 1.0)
        close_location_signed = (close_location - 0.5) * 2.0

        directional_confirmation = 0.6 * close_to_open + 0.4 * close_location_signed

        typical_range = ts_mean(intrabar_range.fillna(0.0), 20)
        range_weight = intrabar_range / (typical_range + eps)
        range_weight = range_weight.fillna(0.0).clip(lower=0.0)

        raw_signal = directional_confirmation * range_weight
        return rank(raw_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0))
