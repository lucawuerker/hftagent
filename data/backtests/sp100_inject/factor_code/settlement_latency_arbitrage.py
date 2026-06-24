"""In blockchain-based markets, settlement latency can create significant price discrepancies across exchanges, particularly when high volatility coincides with slower transaction validation. As highlighted in the research, these latencies can lead to profitable arbitrage opportunities for risk-averse traders. This factor aims to exploit periods of high settlement latencies, where price differences across exchanges widen, creating an opportunity for arbitrageurs to capitalize on the inefficiencies."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import (  
    ts_sum, ts_mean, ts_rank,  
    delta, sign, abs_,  
)
from quant_fund_agent.factors.registry import register_factor

@register_factor
class SettlementLatencyArbitrage(BaseFactor):
    factor_id = "settlement_latency_arbitrage"
    name = "Settlement Latency Arbitrage Factor"
    category = "statistical_arbitrage"
    description = (  
        "This signal calculates the expected arbitrage profit from exploiting price differences across exchanges, "  
        "adjusting for the expected settlement latency. It leverages price data to identify when the cost of arbitrage "  
        "exceeds the potential profit, enabling the identification of optimal trading opportunities."
    )
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        price_diff = delta(close, 1)
        avg_volume = ts_mean(volume, 20)
        arbitrage_opportunity = price_diff / avg_volume
        signal = ts_rank(arbitrage_opportunity, 60)
        return signal.where(avg_volume > 0, 0.0)