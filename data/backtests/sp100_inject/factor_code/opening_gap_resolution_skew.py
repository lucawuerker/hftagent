"""Large overnight or session-opening gaps often reflect information that is only partially digested at the open, when liquidity is thin and price impact is high. If the bar then closes materially back toward the prior open level, it suggests the initial move was overextended and that short-horizon order flow is reverting as the market re-prices the gap. This is complementary to range-based reversal signals because it focuses specifically on the path from open to close relative to the opening dislocation, not just the intrabar extremes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import delay, abs_, sign


@register_factor
class OpeningGapResolutionSkew(BaseFactor):
    """Mean-reversion signal from opening gap size and its intrabar resolution."""

    factor_id = "opening_gap_resolution_skew"
    name = "Opening Gap Resolution Skew"
    category = "mean_reversion"
    description = (
        "Compute the signed opening gap (open vs prior close) and scale it by "
        "the bar's open-to-close recovery toward the prior close. The signal is "
        "strongest when a large gap is partly or fully filled by the close."
    )
    window_length = 2
    inputs = ["open", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        close = data["close"]
        prev_close = delay(close, 1)

        safe_prev_close = prev_close.replace(0, np.nan)
        safe_open = open_.replace(0, np.nan)

        gap_ret = (open_ - safe_prev_close) / safe_prev_close
        recovery = (close - open_) / safe_open

        gap_mag = abs_(gap_ret)
        gap_dir = sign(gap_ret)

        fill_ratio = (close - open_) / (safe_prev_close - open_)
        fill_ratio = fill_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fill_ratio = fill_ratio.clip(lower=-1.0, upper=1.0)

        persistent = (1.0 - fill_ratio).clip(lower=-1.0, upper=1.0)

        raw = -gap_dir * gap_mag * persistent * (1.0 + abs_(recovery))
        signal = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return signal.reindex(index=close.index, columns=close.columns)
