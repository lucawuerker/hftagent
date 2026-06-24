"""Stocks that finish near the highs of their intrabar range after having traded down from the open often reflect persistent informed buying and underreaction to transient selling pressure. By ranking names on how strongly they recover relative to the size of their full bar range, we isolate stocks with the best odds of follow-through on the next bar while filtering out simple large-range noise. This is complementary to lead-lag style effects because it focuses on who is absorbing pressure now rather than who merely moved first."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RelativeIntrabarRecoveryMomentum(BaseFactor):
    """Relative intrabar recovery signal emphasizing strong closes after weakness."""

    factor_id = "relative_intrabar_recovery_momentum"
    name = "Relative Intrabar Recovery Momentum"
    category = "momentum"
    description = (
        "Compute a cross-sectional score from the normalized distance of close "
        "from low versus high, then weight it by the bar's directional body to "
        "favor closes that recover after intrabar weakness."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        range_ = (high - low).replace(0, np.nan)

        recovery_from_low = (close - low) / range_
        distance_from_high = (high - close) / range_
        normalized_recovery = recovery_from_low - distance_from_high

        body = (close - open_) / range_
        intrabar_recovery = normalized_recovery * (1.0 + body)

        signal = intrabar_recovery.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
