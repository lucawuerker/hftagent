"""The paper argues that skewness arises when consecutive trading periods are linked, so the way the market transitions from open to close should contain information about directional asymmetry rather than just volatility. A stock that persistently closes stronger than it opens after adjusting for its intrabar range is likely being revalued by informed flow or delayed reaction, while the opposite pattern signals distribution and fragile demand. Cross-sectionally, this captures a slow-moving preference asymmetry that can complement pure momentum or mean-reversion signals because it focuses on the open-to-close state transition, not the total daily return."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class OvernightOpenCloseAsymmetry(BaseFactor):
    """Cross-sectional open-to-close asymmetry normalized by intrabar range."""

    factor_id = "overnight_open_close_asymmetry"
    name = "Overnight Open-Close Asymmetry"
    category = "other"
    description = (
        "Compute each bar's normalized open-to-close asymmetry as "
        "(close - open) / (high - low + eps), then smooth it with a short "
        "rolling mean to reduce noise. Rank stocks cross-sectionally; high "
        "values indicate persistent positive close-over-open settlement "
        "relative to intrabar range, low values indicate the reverse."
    )
    window_length = 5
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        eps = 1e-12
        intrabar_range = (high - low).replace(0.0, np.nan)
        asymmetry = (close - open_) / (intrabar_range + eps)
        asymmetry = asymmetry.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        smoothed = ts_mean(asymmetry, 5).fillna(0.0)
        return rank(smoothed)
