"""The price impact of trades is a crucial element of market microstructure, as it reflects how order flow can shift asset prices. Excessive buying or selling can create temporary inefficiencies, leading to potential mean-reversion opportunities. This factor leverages the relationship between trade volume and the resulting price movement to identify assets that may experience corrections due to overextended price impacts from recent trade flows."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, ts_sum, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TradeFlowPriceImpactRatio(BaseFactor):
    factor_id = "trade_flow_price_impact_ratio"
    name = "Trade Flow Price Impact Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of the volume of trades to the absolute price change over a specified window. A higher ratio indicates a greater price impact per unit of volume, suggesting the potential for mean reversion as the market corrects itself."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"]
        price_change = abs_(delta(close, 1))
        total_volume = ts_sum(volume, 5)
        price_impact_ratio = total_volume / price_change.where(price_change != 0, np.nan)
        return price_impact_ratio.fillna(0.0)