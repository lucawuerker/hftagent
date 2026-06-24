"""Large short-horizon price moves that are not supported by consistent intrabar direction tend to fade, especially when the bar closes back near the midpoint after making a substantial excursion. This is a microstructure-friendly reversal idea: a move that is mostly a wick or a transient spike often reflects temporary order-flow imbalance rather than durable information. The locality argument in the LAM paper motivates focusing on nearby-path continuity rather than the single terminal close; here we invert that logic and bet against abrupt paths that fail to sustain local trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import abs_, delay, rank, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LocalTrendConsistencyReversion(BaseFactor):
    """Reversion from large intrabar excursions that fail to persist into the close."""

    factor_id = "local_trend_consistency_reversion"
    name = "Local Trend Consistency Reversion"
    category = "mean_reversion"
    description = (
        "Computes a cross-sectional reversal score from the ratio of intrabar path range "
        "to net close-to-open displacement, penalizing bars where the close is near the "
        "open but the high-low excursion is large. Higher values indicate stronger expected "
        "mean reversion on the next bar."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-8

        intrabar_range = (high - low).abs()
        net_move = (close - open_).abs()
        midpoint = (high + low) * 0.5
        close_near_mid = 1.0 - ((close - midpoint).abs() / (intrabar_range + eps))
        close_near_mid = close_near_mid.clip(lower=0.0, upper=1.0)

        range_to_move = intrabar_range / (net_move + eps)
        range_to_move = range_to_move.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Emphasize abrupt bars where the close fails to confirm the excursion.
        excursion_penalty = range_to_move * close_near_mid

        # Add a small local persistence term so isolated spikes with no follow-through
        # are ranked more negatively than consistent path-like moves.
        close_delta = abs_(close - delay(close, 1)).fillna(0.0)
        local_consistency = ts_mean(close_delta, 3).fillna(0.0)

        raw = excursion_penalty - 0.5 * local_consistency
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Cross-sectional rank so the signal is comparable across names each bar.
        score = rank(raw)

        # Prefer stronger reversals when the most recent move is itself large relative
        # to its short-term history.
        recent_move_strength = ts_rank(abs_(close - open_), 5).fillna(0.5)
        score = score * (0.5 + recent_move_strength)

        return score.astype(float)
