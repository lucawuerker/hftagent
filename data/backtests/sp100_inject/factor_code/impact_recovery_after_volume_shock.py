"""Measure whether a bar’s return per unit volume is extreme and whether the close sits near the bar extreme, then score the next-bar expected reversal strength from the combination of shock size and lack of follow-through. Positive values indicate an overextended high-volume move likely to mean-revert; negative values indicate a high-volume move that remains efficiently absorbed and may persist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delta, rank, returns, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ImpactRecoveryAfterVolumeShock(BaseFactor):
    """Mean-reversion signal for mechanically stretched high-volume price shocks."""

    factor_id = "impact_recovery_after_volume_shock"
    name = "Impact Recovery After Volume Shock"
    category = "microstructure"
    inputs = ["open", "high", "low", "close", "volume"]

    window_length = 20

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12

        bar_range = (high - low).replace(0.0, np.nan)
        close_loc = ((close - low) / bar_range).clip(0.0, 1.0).fillna(0.5)
        extreme_close = (close_loc - 0.5).abs() * 2.0

        bar_return = returns(data).fillna(0.0)
        vol_safe = volume.replace(0.0, np.nan)
        impact = (bar_return.abs() / vol_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        shock_rank = ts_rank(impact, 20).fillna(0.5)
        volume_rank = rank(volume.fillna(0.0)).fillna(0.5)

        follow_through = delta(close, 1).fillna(0.0) / (open_.abs() + eps)
        follow_through_rank = rank(follow_through.fillna(0.0)).fillna(0.5)

        reversal_intensity = (shock_rank * extreme_close * volume_rank).fillna(0.0)
        persistence_intensity = ((1.0 - extreme_close) * follow_through_rank).fillna(0.0)

        signal = reversal_intensity - persistence_intensity
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
