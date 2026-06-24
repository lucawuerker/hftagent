"""When a bar closes near its high on unusually heavy volume, it often reflects persistent buying pressure that was able to absorb available supply throughout the interval; the opposite holds near the low. This combines the market-impact intuition of order-flow imbalance (Cont et al., 2014; Xu et al., 2019) with the idea that volume confirms whether the close was driven by informed or aggressive flow rather than noise. The signal should complement existing range- and close-based alphas by emphasizing conviction only when the close is located at an extreme of the bar's realized range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import ts_mean


@register_factor
class RangeCloseVolumeImbalance(BaseFactor):
    """Range-close location weighted by robust volume surprise."""

    factor_id = "range_close_volume_imbalance"
    name = "Range Close Volume Imbalance"
    category = "microstructure"
    description = (
        "Signed close-location score weighted by a trailing volume surprise proxy. "
        "Positive values indicate closes near the bar high on elevated volume; "
        "negative values indicate closes near the bar low on elevated volume."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        range_ = (high - low).replace(0.0, np.nan)
        close_location = ((2.0 * close - high - low) / range_).clip(-1.0, 1.0)

        trailing_vol = ts_mean(volume, 20)
        volume_surprise = volume / trailing_vol.replace(0.0, np.nan)
        volume_surprise = volume_surprise.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        volume_surprise = volume_surprise.clip(lower=0.0)

        signal = close_location * volume_surprise
        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
