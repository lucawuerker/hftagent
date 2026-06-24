"""Rough-volatility research suggests that short-horizon price dynamics are highly irregular, but the *path* taken within the bar still matters for the next-bar drift. When a stock prints an unusually large high-low range yet settles near one extreme, it often reflects one-sided order-flow pressure that has not been fully arbitraged away by the close, especially when the move is efficient rather than noisy. This factor looks for that combination of large realized excursion and low intrabar mean-reversion, which should complement seed signals focused on close location, reversal, or volume imbalance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RelativeRangeEfficiencySurprise(BaseFactor):
    """Cross-sectional surprise in intrabar range efficiency versus a short rolling baseline."""

    factor_id = "relative_range_efficiency_surprise"
    name = "Relative Range Efficiency Surprise"
    category = "other"
    description = (
        "Compute intrabar range efficiency as absolute close-open return divided by high-low range, "
        "then compare it cross-sectionally to a short rolling baseline of the same stock."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        denom = (high - low).abs().replace(0, np.nan)
        net_move = (close - open_).abs()
        efficiency = net_move.divide(denom).replace([np.inf, -np.inf], np.nan)
        efficiency = efficiency.fillna(0.0)

        baseline = ts_mean(efficiency, 5).fillna(0.0)
        surprise = efficiency - baseline

        # Encourage names that are unusually efficient relative to the cross-section,
        # while keeping the signal centered and robust to sparse prints.
        cross_sectional = rank(surprise.fillna(0.0))
        centered = (cross_sectional - 0.5) * 2.0

        range_scale = (high - low).abs().divide((open_.abs() + close.abs()) * 0.5 + eps)
        range_scale = range_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_bias = rank(range_scale) - 0.5

        return centered * (1.0 + range_bias)
