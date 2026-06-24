"""Market impact models suggest that the price response to trades is often temporary, with prices eventually reverting to a mean level. By analyzing the ratio of temporary impact to permanent impact from recent trades, we can identify overreactions in price movements that are likely to reverse. This factor capitalizes on the behavioral bias where traders overreact to new information."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_sum, delta, returns
from quant_fund_agent.factors.registry import register_factor


@register_factor
class TemporaryImpactRatio(BaseFactor):
    factor_id = "temporary_impact_ratio"
    name = "Temporary Impact Ratio"
    category = "microstructure"
    description = ("This signal computes the ratio of the instantaneous price impact from recent trades to the overall price change observed over a defined period. A high ratio indicates potential price overreaction, while a low ratio suggests stability.")
    inputs = ["close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].fillna(0.0)
        volume = data["volume"].fillna(0.0)

        # Calculate the price change over the last 5 bars
        price_change = delta(close, 5)

        # Calculate the temporary impact as the sum of volume over the last 5 bars
        temporary_impact = ts_sum(volume, 5)

        # Calculate the ratio of temporary impact to price change
        impact_ratio = temporary_impact / price_change.replace(0, np.nan)

        return impact_ratio