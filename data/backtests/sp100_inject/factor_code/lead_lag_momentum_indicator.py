"""The lead-lag effect observed in stock markets suggests that certain stocks tend to exhibit momentum based on the performance of leading stocks. By analyzing the returns of a leading stock and its corresponding lagging stock over a defined period, we can capture a momentum signal that exploits this predictive relationship, enhancing potential returns. This approach is informed by research indicating that understanding these dynamics can improve investment strategies.

This factor computes the relative momentum of lagging stocks based on the price movement of their leading counterparts over a specified lag period."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_rank, adv
from quant_fund_agent.factors.registry import register_factor


@register_factor
class LeadLagMomentumIndicator(BaseFactor):
    factor_id = "lead_lag_momentum_indicator"
    name = "Lead-Lag Momentum Indicator"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        lagged_close = close.shift(5)  # Assuming a 5-bar lag for the leading stock
        momentum = delta(close, 5)  # 5-bar momentum of the leading stock
        active = ts_rank(momentum, 60)  # 60-bar rank of the momentum
        fallback = pd.DataFrame(0.0, index=close.index, columns=close.columns)  # Fallback to 0
        return active.where(adv20 < volume, fallback)