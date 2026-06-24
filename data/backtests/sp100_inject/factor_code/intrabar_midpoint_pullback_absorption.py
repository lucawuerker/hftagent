"""When price extends away from the open but closes back toward the bar midpoint, it often reflects aggressive directional flow meeting latent liquidity and getting absorbed before the bar ends. That pattern can be a short-horizon reversal signal because the initiating side spent volume without securing follow-through, leaving the close relatively “undecided.” The intuition is analogous to the paper’s differential-view idea: the path matters, not just the endpoint, and a smooth return toward equilibrium after an excursion often signals inefficient overshoot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarMidpointPullbackAbsorption(BaseFactor):
    """Intrabar reversal signal from failed extensions back toward the bar midpoint."""

    factor_id = "intrabar_midpoint_pullback_absorption"
    name = "Intrabar Midpoint Pullback Absorption"
    category = "microstructure"
    description = (
        "Measure the intrabar excursion from open to high/low relative to the close's "
        "position inside the bar, and score bars where the close sits near the midpoint "
        "after a large directional move. Combine this with volume to emphasize "
        "high-participation failed extensions and produce a cross-sectional reversal signal."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"].fillna(0.0)

        bar_range = (high - low).replace(0, np.nan)
        upper_excursion = (high - open_).clip(lower=0.0)
        lower_excursion = (open_ - low).clip(lower=0.0)
        total_excursion = upper_excursion + lower_excursion

        # Close position within the bar, centered at 0.5, mapped to [-1, 1].
        close_location = ((close - low) / bar_range).clip(lower=0.0, upper=1.0)
        midpoint_proximity = 1.0 - (close_location - 0.5).abs() * 2.0
        midpoint_proximity = midpoint_proximity.clip(lower=0.0, upper=1.0)

        # Directional extension away from the open, normalized by the bar range.
        directional_bias = ((close - open_) / bar_range).clip(lower=-1.0, upper=1.0)
        extension = (total_excursion / bar_range).replace([np.inf, -np.inf], np.nan)
        extension = extension.fillna(0.0)

        # Failed extension is strongest when the bar ranges far but ends near the midpoint.
        failed_extension = extension * midpoint_proximity

        # High-participation bars should matter more, but with a robust scale to avoid blow-ups.
        vol_scale = volume.rolling(20, min_periods=1).mean().replace(0, np.nan)
        participation = (volume / vol_scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        participation = participation.clip(lower=0.0)

        # Reversal sign: positive when the bar extends up then fails, negative when it extends down then fails.
        reversal_sign = -directional_bias

        signal = failed_extension * participation * reversal_sign

        # Damp signals from tiny bars and enforce a stable numeric output.
        active = (bar_range > 0).astype(float)
        signal = signal * active
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return signal
