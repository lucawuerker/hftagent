"""The concept of minimum-energy control suggests that asset prices tend to gravitate towards a state that requires the least effort to maintain, reflecting underlying market dynamics. This factor leverages the idea that prices will revert to equilibrium levels following significant deviations, as described in the paper 'Learning Minimum-Energy Controls from Heterogeneous Data'. By tracking how much energy (or trading volume) is required to maintain a price level, we can identify potential price reversals and momentum opportunities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (delta, ts_sum, ts_mean, adv, sign, abs_, ts_rank)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class MinimumEnergyControlPriceSignal(BaseFactor):
    factor_id = "minimum_energy_control_price_signal"
    name = "Minimum Energy Control Price Signal"
    category = "momentum"
    description = "This factor computes the required energy to drive the price towards its equilibrium state based on volume and price changes over time. It uses the relationship between price movements and trading volumes to establish a control signal indicating potential price reversion."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        price_change = delta(close, 1)
        energy = abs_(price_change) * volume
        energy_signal = ts_sum(energy, 5)
        control_signal = ts_rank(energy_signal, 60) * sign(price_change)
        fallback = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        return control_signal.where(adv20 > volume, fallback)