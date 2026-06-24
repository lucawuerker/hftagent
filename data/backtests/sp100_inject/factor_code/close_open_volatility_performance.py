"""When a bar closes near its open despite having traded through a wide intrabar range, it often indicates that volatility was absorbed rather than converted into directional price discovery. In practice, that can reflect transient overreaction, dealer hedging flows, or liquidity provision stepping in late in the bar, which tends to be followed by softer subsequent moves. This idea is consistent with the paper’s broader theme that persistent shape/level dynamics can be forecast when the market’s current move is more about temporary pressure than durable information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import adv, rank, ts_mean


@register_factor
class CloseOpenVolatilityPerformance(BaseFactor):
    """Close-to-open compression versus intrabar range, with volume emphasis."""

    factor_id = "close_open_volatility_performance"
    name = "Close-Open Volatility Performance"
    category = "volatility"
    description = (
        "High when the close-to-open body is small relative to the high-low range, "
        "especially after a wide excursion and on heavier volume."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        body = (close - open_).abs()
        range_ = (high - low).abs()

        safe_body = body.replace(0, np.nan)
        safe_range = range_.replace(0, np.nan)

        ratio = safe_range / safe_body
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        range_score = rank(ts_mean(ratio, 5))
        volume_score = rank(volume / adv(volume, 20).replace(0, np.nan))
        compression_score = rank((safe_range / (safe_body + safe_range)).replace([np.inf, -np.inf], np.nan).fillna(0.0))

        signal = (range_score * 0.5) + (volume_score * 0.3) + (compression_score * 0.2)
        return signal.fillna(0.0)
