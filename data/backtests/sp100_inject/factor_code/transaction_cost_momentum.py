"""In markets with significant transaction costs, price movements can exhibit momentum due to traders' reluctance to enter trades when costs are high, leading to prolonged trends. According to the paper by Cordero and Perez-Ostafe, trading strategies can become asymptotically arbitrage opportunities under conditions of low transaction costs, suggesting that momentum can be derived from price changes in relation to these costs.

This factor computes the momentum of price changes adjusted for recent trading volume and transaction costs, indicating the potential for continued price movement in the same direction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (rank, delta, ts_rank, adv)
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TransactionCostMomentum(BaseFactor):
    factor_id = "transaction_cost_momentum"
    name = "Transaction Cost Momentum"
    category = "momentum"
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"].fillna(0.0)
        adv20 = adv(volume, 20)
        price_change = delta(close, 1)
        momentum_signal = ts_rank(price_change, 60)
        return momentum_signal.where(adv20 > volume, -1.0)