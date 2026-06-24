"""A bar that trades down hard but closes back near its high often reflects failed sell pressure and hidden demand absorbing inventory. In market microstructure terms, this resembles a brief liquidity shock that gets repaired before the close, which is typically more bullish than a comparable bar that stays weak into the close. The idea is consistent with the paper's emphasis that simple state-dependent rules can adapt to varying volatility and still approximate optimal risk control."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class BarRecoveryResilience(BaseFactor):
    """Measure intrabar recovery from downside pressure relative to range."""

    factor_id = "bar_recovery_resilience"
    name = "Bar Recovery Resilience"
    category = "other"
    description = (
        "Measure how far the close sits above the bar midpoint relative to the full high-low range, "
        "and multiply by the normalized downside excursion from open to low. Higher values indicate "
        "stronger intrabar recovery from downside pressure."
    )
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        midpoint = 0.5 * (high + low)

        recovery = (close - midpoint) / bar_range
        downside_excursion = (open_ - low) / bar_range

        signal = recovery * downside_excursion
        signal = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
