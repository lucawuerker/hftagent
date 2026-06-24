"""In Hawkes-style markets, clustered activity tends to persist because trades and quote updates beget more trades in the same direction before the excitation decays. When the bar closes near its extreme on elevated volume, that often reflects an ongoing local order-flow cascade rather than a fully exhausted move, especially if the close is not yet overextended relative to the bar’s own range. This factor tries to capture that short-horizon continuation effect using only OHLCV, complementing pure range- or volume-based reversals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarIntensityPersistence(BaseFactor):
    """Intrabar close-location and volume persistence signal."""

    factor_id = "intrabar_intensity_persistence"
    name = "Intrabar Intensity Persistence"
    category = "microstructure"
    description = (
        "Measures whether a bar’s close is both close to the high/low and supported by above-normal volume, "
        "then smooths it over a short lookback to estimate persistence of directional intensity. "
        "Positive values indicate bullish continuation pressure; negative values indicate bearish continuation pressure."
    )
    window_length = 8
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rng = (high - low).replace(0, np.nan)
        close_pos = ((close - low) / rng).clip(lower=0.0, upper=1.0)
        upper_wick = ((high - close) / rng).clip(lower=0.0, upper=1.0)
        lower_wick = ((close - low) / rng).clip(lower=0.0, upper=1.0)

        body = close - open_
        body_ratio = (body.abs() / rng).clip(lower=0.0, upper=1.0)
        direction = np.sign(body)

        vol_rel = volume / adv(volume, 20)
        vol_rel = vol_rel.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        bullish_intensity = close_pos * body_ratio * vol_rel * (1.0 - upper_wick)
        bearish_intensity = lower_wick * body_ratio * vol_rel * (1.0 - close_pos)

        raw = (bullish_intensity - bearish_intensity) * direction
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        persistence = ts_mean(raw, 5)
        momentum_bias = rank(close - open_) - 0.5
        output = persistence + 0.25 * momentum_bias
        return output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
