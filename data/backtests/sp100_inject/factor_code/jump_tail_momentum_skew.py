"""Cont and Tankov emphasize that jumps create the economically important part of short-horizon risk, and that markets often price in asymmetric jump fear rather than just smooth diffusion. This factor looks for names that exhibit persistent upside-dominant intrabar tail moves after normalizing by recent volatility: when the bar’s upper tail repeatedly dominates the lower tail, the path is often being repriced upward through discrete shocks rather than slow drift. Such patterns can continue for a short horizon because jump clustering and underreaction to information shocks tend to produce local continuation after the initial discontinuity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, ts_mean, stddev
from quant_fund_agent.factors.registry import register_factor


@register_factor
class JumpTailMomentumSkew(BaseFactor):
    """Short-horizon continuation signal from upside-dominant intrabar tail imbalance."""

    factor_id = "jump_tail_momentum_skew"
    name = "Jump Tail Momentum Skew"
    category = "other"
    description = (
        "Compute a cross-sectional score from recent signed tail imbalance: "
        "recent average of (high-close)/(high-low) minus (close-low)/(high-low), "
        "normalized by recent realized volatility and volume. Higher values indicate "
        "repeated upward jump dominance relative to downward tail pressure."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        bar_range = (high - low).replace(0, np.nan)
        upper_tail = (high - close) / bar_range
        lower_tail = (close - low) / bar_range
        tail_imbalance = upper_tail - lower_tail

        tail_avg = ts_mean(tail_imbalance, 5)
        vol_realized = stddev(close.pct_change().fillna(0.0), 10)
        vol_realized = vol_realized.replace(0, np.nan)

        vol_avg = adv(volume.fillna(0.0), 10)
        vol_avg = vol_avg.replace(0, np.nan)

        score = tail_avg / vol_realized
        score = score / np.log1p(vol_avg)
        score = score.replace([np.inf, -np.inf], np.nan)
        return score.fillna(0.0)
