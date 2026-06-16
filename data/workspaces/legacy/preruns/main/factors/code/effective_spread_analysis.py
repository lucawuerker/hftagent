"""The effective spread serves as a crucial measure of transaction costs, encapsulating both the bid-ask spread and the price impact of trades. According to Markov's framework, understanding the dynamics of effective spreads can help traders gain insights into liquidity conditions and the potential for adverse selection, which can inform entry and exit strategies. By monitoring changes in effective spreads, traders can identify optimal trading moments and avoid costly trades in illiquid conditions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, rank

@register_factor
class EffectiveSpreadAnalysis(BaseFactor):
    factor_id = "effective_spread_analysis"
    name = "Effective Spread Analysis"
    category = "microstructure"
    inputs = ["effSpread", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the average effective spread over a 60-bar window
        avg_eff_spread = ts_mean(eff_spread, 60)

        # Calculate the total volume over the same window
        total_volume = ts_sum(volume, 60)

        # Normalize the effective spread by the total volume to get a signal
        signal = avg_eff_spread / total_volume.replace(0, np.nan)

        # Rank the signal to get a relative measure
        ranked_signal = rank(signal)

        return ranked_signal