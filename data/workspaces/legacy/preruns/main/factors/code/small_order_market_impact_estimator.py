"""The market impact of small orders can provide insight into the behavior of informed traders, as highlighted in the paper 'Market Impact of Small Orders'. By analyzing the signed trade volumes and their subsequent price movements, we can estimate the average participation rate of informed trading activity. This factor serves to capture the incremental price changes associated with small trades, which can signal potential upcoming price movements."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_sum, ts_mean, sign, abs_, delay


@register_factor
class SmallOrderMarketImpactEstimator(BaseFactor):
    factor_id = "small_order_market_impact_estimator"
    name = "Small Order Market Impact Estimator"
    category = "microstructure"
    inputs = ["trade", "volume", "effSpread", "effLobImb"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        trade = data["trade"].fillna(0.0)
        volume = data["volume"].fillna(0.0)
        eff_spread = data["effSpread"].fillna(0.0)
        eff_lob_imb = data["effLobImb"].fillna(0.0)

        # Calculate the signed trade volume over the last 5 bars
        signed_trade_sum = ts_sum(trade, 5)
        total_volume_sum = ts_sum(volume, 5)

        # Avoid division by zero by replacing zeros with NaN
        total_volume_sum = total_volume_sum.replace(0, np.nan)

        # Calculate the average market impact of small orders
        market_impact = signed_trade_sum / total_volume_sum

        # Normalize the market impact by the effective spread and LOB imbalance
        market_impact_normalized = market_impact * (eff_spread + eff_lob_imb)

        # Return the result, ensuring it has the same index and columns as close
        return market_impact_normalized.where(~market_impact_normalized.isna(), 0.0)