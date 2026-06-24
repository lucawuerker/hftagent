"""The paper argues that arbitrageurs care about both price risk and the time it takes to complete settlement; when trading frictions are high, mispricings persist because capital cannot be deployed fast enough. In a single-asset intraday setting, a bar that travels a large fraction of its high-low range but finishes near the open suggests that the move was probed and then reversed, consistent with transient liquidity shocks that could not be cleanly arbitraged away. Such bars should tend to exhibit short-horizon mean reversion, especially when the bar’s range is wide relative to the net open-to-close move."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeSettlementEfficiency(BaseFactor):
    """Inverted intrabar settlement efficiency signal emphasizing wide-range reversals."""

    factor_id = "range_settlement_efficiency"
    name = "Range Settlement Efficiency"
    category = "microstructure"
    description = (
        "Compute the signed efficiency of each bar as the close-to-open change "
        "divided by the intrabar high-low range, then invert it so that bars "
        "with large excursion but weak directional follow-through score highest. "
        "Cross-sectionally, this identifies tickers whose current bar looks most "
        "like an inefficient price discovery episode rather than a persistent trend."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        bar_range = (high - low).replace(0.0, np.nan)
        net_move = close - open_

        signed_efficiency = net_move / bar_range
        inverted_efficiency = -signed_efficiency

        # Favor bars with large excursion but weak net direction, while staying stable
        # around zero-range or fully missing bars.
        signal = inverted_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return signal
