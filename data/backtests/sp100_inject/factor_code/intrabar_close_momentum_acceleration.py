"""Markets often exhibit short-horizon continuation when price closes near the bar high after repeatedly improving during the bar, because aggressive flow is still dominating and liquidity providers have not yet fully replenished. This idea is inspired by the ECG paper’s emphasis on capturing transient signal shape and local dynamics with wavelets: here we measure whether recent bars are not just trending, but trending with accelerating close-to-close progress inside the bar. The signal should be strongest when the latest close improves relative to the prior close, while the bar itself also closes strong versus its own open and range."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import delta, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class IntrabarCloseMomentumAcceleration(BaseFactor):
    """Short-horizon momentum enhanced by intrabar close-strength acceleration."""

    factor_id = "intrabar_close_momentum_acceleration"
    name = "Intrabar Close Momentum Acceleration"
    category = "momentum"
    description = (
        "Compute a cross-sectional score from a short average of close-to-close "
        "returns multiplied by a normalized closing-position term, with a recent "
        "acceleration component from the change in that score."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_close = close.shift(1)
        ret = (close / prev_close) - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        bar_range = (high - low).replace(0.0, np.nan)
        close_pos = ((close - open_) / bar_range).replace([np.inf, -np.inf], np.nan)
        close_pos = close_pos.fillna(0.0).clip(lower=-1.0, upper=1.0)

        short_momentum = ts_mean(ret, 3)
        close_strength = ts_mean(close_pos, 3)

        base_signal = short_momentum * close_strength
        accel = delta(base_signal, 1)

        score = base_signal + 0.5 * accel
        score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        cs_momentum = rank(score)
        cs_accel = rank(accel.replace([np.inf, -np.inf], np.nan).fillna(0.0))

        out = (0.7 * cs_momentum) + (0.3 * cs_accel)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
