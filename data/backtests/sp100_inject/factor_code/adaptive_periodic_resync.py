"""Markets often drift away from short-horizon equilibrium and then snap back when the current bar re-anchors to the prior intrabar structure. Inspired by the paper’s point that affine mappings work best when a recent window contains a usable “period,” this factor looks for cases where the latest close realigns with the bar’s normalized open-to-close path after a temporary excursion. Strong resynchronization after a misalignment is often a sign that the move was transitory rather than directional, creating a short-horizon reversal edge."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, stddev, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class AdaptivePeriodicResync(BaseFactor):
    """Adaptive periodic resynchronization signal from bar-location alignment."""

    factor_id = "adaptive_periodic_resync"
    name = "Adaptive Periodic Resynchronization"
    category = "other"
    description = (
        "Compute the sign-adjusted alignment between the current close location "
        "within the bar and the prior bar’s close location, normalized by bar "
        "range and filtered by recent volatility. Positive values indicate the "
        "bar has re-synchronized with the recent local pattern; negative values "
        "indicate persistent desynchronization."
    )
    window_length = 30
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        bar_range = (high - low).replace(0, np.nan)
        close_loc = (close - low) / bar_range
        open_loc = (open_ - low) / bar_range

        prior_close_loc = close_loc.shift(1)
        prior_open_loc = open_loc.shift(1)

        # Alignment improves when the current close re-anchors relative to the
        # previous intrabar path and the current position is above the prior
        # bar's close location after a temporary excursion.
        alignment = (close_loc - prior_close_loc) - 0.5 * (open_loc - prior_open_loc)
        resync = close_loc - prior_close_loc
        excursion = (close - open_).abs() / bar_range

        # Recent volatility filter: suppress unstable, noisy regimes.
        vol = stddev(close.pct_change(), 10).fillna(0.0)
        vol_rank = ts_rank(vol, 10).fillna(0.0)

        # Strengthen signals only when the bar shows a meaningful normalized move
        # and recent volatility is not excessive.
        alignment_smooth = ts_mean(alignment, 3).fillna(0.0)
        resync_smooth = ts_mean(resync, 3).fillna(0.0)
        excursion_smooth = ts_mean(excursion, 3).fillna(0.0)

        score = (alignment_smooth + 0.5 * resync_smooth) * (1.0 + excursion_smooth)
        score = score * (1.0 - 0.7 * vol_rank)

        # If the latest bar is still materially desynchronized from the prior
        # bar's location, penalize the signal to favor reversal instead of trend.
        desync_penalty = ts_rank((prior_close_loc - close_loc).abs(), 5).fillna(0.0)
        score = score * (1.0 - 0.5 * desync_penalty)

        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return score
