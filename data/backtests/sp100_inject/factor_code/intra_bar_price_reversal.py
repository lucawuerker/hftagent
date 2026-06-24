"""Intra-bar price movements often reflect temporary market inefficiencies. A significant price reversal within the trading session can indicate that the market is correcting itself, allowing traders to capitalize on short-term fluctuations. This aligns with the behavioral finance concept that traders often overreact to news, leading to price reversals as they reassess their positions."""

from __future__ import annotations

import pandas as pd
import numpy as np

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import ts_max, ts_min, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntraBarPriceReversal(BaseFactor):
    factor_id = "intra_bar_price_reversal"
    name = "Intra-Bar Price Reversal Indicator"
    category = "mean_reversion"
    description = ("This signal computes the difference between the high and low prices within a bar, "
                   "normalized by the bar's open price. A large reversal signal indicates a potential "
                   "mean reversion opportunity.")
    inputs = ["open", "high", "low"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_price = data["open"].fillna(0.0)
        high_price = data["high"].fillna(0.0)
        low_price = data["low"].fillna(0.0)

        price_reversal = (ts_max(high_price, 1) - ts_min(low_price, 1)) / open_price
        return price_reversal.where(open_price != 0, 0.0)