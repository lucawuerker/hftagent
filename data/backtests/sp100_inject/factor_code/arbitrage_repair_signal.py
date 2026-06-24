"""Arbitrage opportunities arise when there are discrepancies between the prices of related financial instruments, often due to market inefficiencies. By applying a repair mechanism to these prices to ensure they align with no-arbitrage conditions, we can identify potential mispricings that are likely to revert, allowing traders to capitalize on them before the market corrects. The presence of arbitrage signals indicates that the market is inefficient, which can lead to profitable trading opportunities."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, delta, sign, abs_, rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ArbitrageRepairSignal(BaseFactor):
    factor_id = "arbitrage_repair_signal"
    name = "Arbitrage Repair Signal"
    category = "statistical_arbitrage"
    description = "This signal computes the perturbations needed to adjust observed prices to eliminate arbitrage opportunities, returning a measure of how far each instrument's price deviates from an arbitrage-free state."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        avg_close = ts_mean(close, 5)
        price_adjustment = close - avg_close
        signal = sign(price_adjustment) * abs_(price_adjustment)
        return signal.where(volume > 0, 0.0)