"""The implicit spread is a critical measure for large tick assets that serves as a proxy for the bid-ask spread when traditional measures fail. By analyzing the ratio of the implicit spread to the realized volatility, we can identify periods where market participants may be underpricing risk or overestimating liquidity, leading to potential trading opportunities as prices adjust to reflect true market conditions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, stddev, delay, vwap
from quant_fund_agent.factors.registry import register_factor


@register_factor
class ImplicitSpreadVolatilityRatio(BaseFactor):
    factor_id = "implicit_spread_volatility_ratio"
    name = "Implicit Spread Volatility Ratio"
    category = "microstructure"
    description = "This factor computes the ratio of the implicit spread (defined as ηα, where η is the microstructure parameter and α is the tick size) to the realized volatility over a specified period. It highlights mispricings in large tick assets, allowing traders to capitalize on potential reversals or momentum."
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        volume = data["volume"].fillna(0.0)

        # Calculate the implicit spread (ηα) as a function of volume
        implicit_spread = vwap(data) * volume

        # Calculate realized volatility over a 5-bar window
        realized_volatility = stddev(close, 5)

        # Calculate the ratio of implicit spread to realized volatility
        ratio = implicit_spread / realized_volatility.replace(0, np.nan)

        # Return the ratio with the same index and columns as close
        return ratio