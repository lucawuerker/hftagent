"""When a bar makes a large excursion but fails to hold that move into the close, it often reflects informed liquidity taking the other side of an overextended push. This factor targets names where the close is near the bar's start despite substantial intrabar range and volume, indicating inefficient price discovery and a likely short-horizon reversal. The idea is consistent with the paper's emphasis on scenario-dependent risk sharing: failed absorption shocks are informative about transient mispricing rather than persistent fundamental repricing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import rank, ts_mean, adv, abs_, delta


@register_factor
class CloseReversalEfficiencyGap(BaseFactor):
    """Score bars where intrabar effort is large but close-to-open progress is weak."""

    factor_id = "close_reversal_efficiency_gap"
    name = "Close Reversal Efficiency Gap"
    category = "mean_reversion"
    description = (
        "Cross-sectional score that is high when a bar shows large range "
        "and volume but a small net close relative to the open."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        eps = 1e-12

        bar_range = (high - low).abs()
        net_move = (close - open_).abs()

        range_efficiency_gap = (bar_range - net_move) / (bar_range + eps)
        range_efficiency_gap = range_efficiency_gap.clip(lower=0.0)

        rel_range = bar_range / (open_.abs() + eps)
        rel_volume = volume / (adv(volume, 20).replace(0, np.nan))
        rel_volume = rel_volume.fillna(0.0)

        short_reversal = (open_ - close).abs() / (bar_range + eps)
        short_reversal = short_reversal.clip(lower=0.0, upper=1.0)

        efficient_gap = rank(range_efficiency_gap)
        range_score = rank(rel_range)
        volume_score = rank(ts_mean(rel_volume, 5))
        reversal_score = rank(short_reversal)

        raw = (0.40 * efficient_gap) + (0.25 * range_score) + (0.20 * volume_score) + (0.15 * reversal_score)

        direction = -np.sign(close - open_)
        direction = pd.DataFrame(direction, index=close.index, columns=close.columns).fillna(0.0)

        final = raw * direction
        final = final.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return final
