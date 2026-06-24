"""Large candles that close near the edge of the bar are more informative when they occur on elevated volume, because they are harder to explain as random noise or thin-liquidity prints. In contrast, a strong close with weak participation is often prone to fade once liquidity replenishes. This factor looks for short-horizon continuation when price displacement and traded participation reinforce each other, complementing pure imbalance or reversal signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, abs_, sign
from quant_fund_agent.factors.registry import register_factor


@register_factor
class BodyVolumeConfirmation(BaseFactor):
    """Signed candle body scaled by normalized volume surprise."""

    factor_id = "body_volume_confirmation"
    name = "Body-Volume Confirmation"
    category = "microstructure"
    description = (
        "Compute the signed candle body relative to intrabar range, then scale "
        "it by a normalized volume surprise versus recent volume. High positive "
        "values indicate bullish displacement with unusually strong participation; "
        "high negative values indicate bearish displacement with unusually strong participation."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        body = close - open_
        rng = (high - low).replace(0, np.nan)
        body_ratio = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        vol20 = adv(volume, 20)
        vol_surprise = (volume / vol20.replace(0, np.nan)) - 1.0
        vol_surprise = vol_surprise.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        participation = sign(vol_surprise) * abs_(vol_surprise)
        signal = body_ratio * (1.0 + participation)

        return signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
