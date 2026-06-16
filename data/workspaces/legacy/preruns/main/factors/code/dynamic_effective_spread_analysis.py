"""The effective spread captures the true cost of trading and reflects market liquidity. By analyzing the dynamic changes in the effective spread over time, we can identify periods of heightened market activity or stress, which may precede price movements. This is particularly relevant in the context of the Hawkes process, as it accounts for self-exciting behavior in trading volume that can lead to temporary price impacts."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_mean, ts_sum, ts_rank, abs_, delay
from quant_fund_agent.factors.registry import register_factor


@register_factor
class DynamicEffectiveSpreadAnalysis(BaseFactor):
    factor_id = "dynamic_effective_spread_analysis"
    name = "Dynamic Effective Spread Analysis"
    category = "microstructure"
    inputs = ["effSpread", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        eff_spread = data["effSpread"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate rolling average of effective spread over a window of 60 bars
        rolling_avg_eff_spread = ts_mean(eff_spread, 60)

        # Adjust for changes in trading volume
        adjusted_signal = rolling_avg_eff_spread.where(volume > 0, np.nan)

        # Return the adjusted signal
        return adjusted_signal
