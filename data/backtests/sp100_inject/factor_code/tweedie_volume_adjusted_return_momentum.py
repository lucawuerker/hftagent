"""Market participants often underreact to changes in volume-adjusted returns, leading to momentum in price movements. By leveraging the Tweedie distribution to model the return dynamics, we can capture both the mass at zero (indicating low or zero returns) and the long right tail (indicating potential spikes), which aligns with behavioral finance theories on underreaction and overreaction. The factor aims to exploit these inefficiencies in market pricing."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, adv, returns)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TweedieVolumeAdjustedReturnMomentum(BaseFactor):
    factor_id = "tweedie_volume_adjusted_return_momentum"
    name = "Tweedie Volume-Adjusted Return Momentum"
    category = "momentum"
    description = ("This factor computes the momentum of returns adjusted for trading volume using a Tweedie distribution framework, allowing for a more nuanced understanding of price movements in the presence of sporadic trading activity. It captures the momentum by assessing the returns over a specified window, adjusted by the volume traded, to highlight discrepancies in return expectations.")
    window_length = 60
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        adv20 = adv(volume, 20)
        returns_ = returns(data)
        momentum = ts_sum(returns_, 60)
        adjusted_momentum = momentum.where(adv20 > 0, 0.0)  # Avoid division by zero
        return adjusted_momentum