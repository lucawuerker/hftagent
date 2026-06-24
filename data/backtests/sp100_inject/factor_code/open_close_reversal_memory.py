"""Large intrabar moves that fail to hold from open to close often reflect temporary price impact rather than durable information. When the bar closes back toward the open after exploring a wide range, it suggests liquidity provision absorbed directional pressure and that the move may partially reverse on the next bar. This is consistent with the expectation-constraint/control view in the cited paper: short-horizon actions are bounded by state constraints and often revert once the control effort is exhausted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import rank, sign


@register_factor
class OpenCloseReversalMemory(BaseFactor):
    """Open-to-close reversal memory scaled by range and bar-end location."""

    factor_id = "open_close_reversal_memory"
    name = "Open-Close Reversal Memory"
    category = "mean_reversion"
    description = (
        "Compute a signed reversal score from the open-to-close return scaled "
        "by the intrabar range, then amplify it when the close sits near the "
        "opposite side of the bar relative to the direction of the open-to-close move."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        bar_range = (high - low).abs().replace(0.0, np.nan)
        oc_return = (close - open_) / open_.replace(0.0, np.nan)
        range_scaled_move = oc_return / bar_range.replace(0.0, np.nan)

        close_from_high = (high - close) / bar_range
        close_from_low = (close - low) / bar_range
        close_location = close_from_low - close_from_high

        reversal_core = -sign(range_scaled_move) * np.abs(range_scaled_move)
        opposite_side = 1.0 + close_location.abs()
        memory_signal = reversal_core * opposite_side

        raw = memory_signal.where(np.isfinite(memory_signal), 0.0)
        cs_rank = rank(raw)
        return (2.0 * cs_rank - 1.0).where(np.isfinite(cs_rank), 0.0)
