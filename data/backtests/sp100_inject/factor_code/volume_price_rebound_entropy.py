"""When price extends to an intrabar extreme on unusually high volume, the move is often driven by temporary order-flow imbalance rather than durable information. If the close then finishes back toward the bar midpoint, it suggests the market absorbed the shock and failed to convert trading intensity into trend persistence, a classic short-horizon reversal setup. The factor is designed to complement pure wick- or range-based reversal signals by explicitly conditioning on whether the move was “paid for” by volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class VolumePriceReboundEntropy(BaseFactor):
    """Volume-weighted close-location rejection signal around the bar midpoint."""

    factor_id = "volume_price_rebound_entropy"
    name = "Volume-Price Rebound Entropy"
    category = "statistical_arbitrage"
    description = (
        "Measure the signed distance of the close from the bar midpoint, "
        "scaled by the full range and weighted by relative volume versus a "
        "rolling baseline. Negative values indicate high-volume rejection "
        "after an extreme excursion; positive values indicate acceptance "
        "near the bar edge."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Bar geometry
        bar_mid = (high + low) * 0.5
        bar_range = (high - low).replace(0, np.nan)
        close_mid_dist = (close - bar_mid) / bar_range

        # Signed close location in [ -1, 1 ]-ish, with rejection strongest near extremes.
        close_location = 2.0 * close_mid_dist

        # Detect whether the bar meaningfully explored its range away from the open.
        upward_excursion = (high - open_) / bar_range
        downward_excursion = (open_ - low) / bar_range
        excursion_pressure = (upward_excursion - downward_excursion).fillna(0.0)

        # Relative volume weight versus rolling baseline; guarded for sparse/zero volume.
        vol_baseline = adv(volume, 20)
        vol_ratio = (volume / vol_baseline.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vol_ratio = vol_ratio.fillna(0.0)
        vol_weight = rank(vol_ratio)

        # Rejection is strong when the close is near the midpoint after a large excursion.
        rebound_component = (-close_location) * (1.0 - abs_(close_location)).fillna(0.0)

        # Entropy-style dampener: if volume is ordinary, the signal should compress toward zero.
        volume_surprise = (vol_weight - 0.5) * 2.0
        baseline_trend = ts_mean(close_location.fillna(0.0), 5).fillna(0.0)

        signal = (rebound_component * volume_surprise) + (0.25 * excursion_pressure * baseline_trend)
        signal = signal.where(bar_range.notna(), 0.0)
        signal = signal.replace([np.inf, -np.inf], 0.0)
        return signal.fillna(0.0)
