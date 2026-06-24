"""Market conditions often shift between different regimes, impacting how price reacts to volume changes. By employing a regime-switching framework, we can capture the momentum driven by volume in various market states, enhancing prediction accuracy in trending environments. This aligns with findings from the paper on optimal control in regime-switching environments, which highlights how different dynamics can dictate market behavior."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_mean, adv, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RegimeSwitchingVolumeMomentum(BaseFactor):
    factor_id = "regime_switching_volume_momentum"
    name = "Regime-Switching Volume Momentum"
    category = "momentum"
    inputs = ["close", "volume"]
    description = "This factor computes the momentum of price based on volume changes, adjusted for the identified regime of the market. It uses the volume data along with price momentum to create a signal that indicates the strength of price trends across different market conditions."

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        momentum = delta(close, 5)
        momentum_signal = ts_rank(momentum, 60)
        return momentum_signal.where(volume > adv20, 0.0)