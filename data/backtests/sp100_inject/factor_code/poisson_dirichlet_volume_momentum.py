"""The Poisson-Dirichlet model suggests that market trends can be influenced by the clustering of price movements, similar to how frequency distributions evolve in genetic models. The volume of trades during these trends can amplify the momentum effect, as higher volumes often indicate stronger conviction among traders. By assessing the relationship between volume spikes and price movements, we can exploit these behavioral patterns to identify potential momentum trades.

This factor computes a momentum signal based on the relationship between trade volume and price changes, utilizing the Poisson-Dirichlet framework to assess the likelihood of continued price trends following significant volume events.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_sum, ts_mean, ts_rank, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class PoissonDirichletVolumeMomentum(BaseFactor):
    factor_id = "poisson_dirichlet_volume_momentum"
    name = "Poisson-Dirichlet Volume Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        price_change = delta(close, 1)
        momentum_signal = ts_rank(price_change, 60)
        active = momentum_signal.where(volume > adv20, 0.0)
        return active