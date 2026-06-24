"""When price moves a lot on unusually little volume, the move is often mechanically fragile: it can reflect thin liquidity, temporary order-flow imbalance, or a lack of broad participation rather than durable information. I expect these moves to partially fade over the next bars, especially if the bar closes near the open after briefly stretching the range, because the market has shown poor conviction in sustaining the excursion. This complements trend and breakout signals by isolating moves that look powerful in price but weak in traded effort."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, sign, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumeEfficiencyGapDecay(BaseFactor):
    """Fade large intrabar moves on low volume when the close fails to hold the excursion."""

    factor_id = "volume_efficiency_gap_decay"
    name = "Volume Efficiency Gap Decay"
    category = "microstructure"
    description = (
        "Compute a price-move-per-volume proxy and penalize bars where the "
        "close gives back much of the intrabar excursion, favoring fade "
        "signals on high-move/low-volume bars with weak holding power."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"].fillna(0.0)

        range_ = (high - low).abs()
        bar_move = (close - open_).abs()
        gap = (close - open_)

        # Normalize move by traded effort; protect against zero/NaN volume.
        effort = volume.replace(0, np.nan)
        move_per_vol = bar_move / effort

        # Recovery / failure-to-hold component: positive when close is near open
        # after a large excursion, especially if the bar range was wide.
        recovery = 1.0 - ((close - open_).abs() / (range_.replace(0, np.nan)))
        recovery = recovery.fillna(0.0)

        # Directional fade bias: if the bar closes against the excursion direction,
        # the signal should be stronger. Use signed power to emphasize larger moves.
        direction = -sign(gap)
        magnitude = np.sqrt(bar_move.fillna(0.0))

        # Combine low-volume efficiency with weak holding power.
        raw = move_per_vol.fillna(0.0) * (1.0 + recovery) * (1.0 + magnitude)
        out = raw * direction

        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
