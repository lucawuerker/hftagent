"""When a bar closes near its high on unusually strong volume, it often signals informed buying pressure that is still being underpriced by slower participants. I expect this effect to be strongest when the close-to-open return is positive and the bar’s internal range is efficient, because those moves are more likely to reflect sustained accumulation than transient noise. The paper’s emphasis on using distributional, condition-sensitive signals reinforces the idea that the joint state of price path and activity matters more than any single point estimate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class CloseVolumePriceSprint(BaseFactor):
    """Cross-sectional momentum signal from close location, intrabar return, and volume surprise."""

    factor_id = "close_volume_price_sprint"
    name = "Close-Volume Price Sprint"
    category = "momentum"
    description = (
        "Computes a cross-sectional score that combines the bar’s close "
        "location within the range, positive open-to-close return, and "
        "volume surprise versus a rolling baseline. Higher values indicate "
        "names that are closing strongly on confirmation volume, which is a "
        "continuation-friendly intraday momentum setup."
    )
    window_length = 20
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        price_range = (high - low).replace(0.0, np.nan)
        close_loc = ((close - low) / price_range).clip(0.0, 1.0).fillna(0.5)

        oc_ret = ((close - open_) / open_.replace(0.0, np.nan)).fillna(0.0)
        oc_ret_pos = oc_ret.clip(lower=0.0)

        range_efficiency = ((close - open_).abs() / price_range).clip(0.0, 1.0).fillna(0.0)

        vol_ma = ts_mean(volume, 20)
        vol_surprise = ((volume / vol_ma.replace(0.0, np.nan)) - 1.0).fillna(0.0)
        vol_surprise = vol_surprise.clip(lower=0.0)

        sprint_raw = (
            0.40 * close_loc
            + 0.30 * rank(oc_ret_pos)
            + 0.20 * rank(range_efficiency)
            + 0.10 * rank(vol_surprise)
        )

        confirmation = rank(close_loc * (1.0 + oc_ret_pos) * (1.0 + vol_surprise))
        signal = 0.6 * sprint_raw + 0.4 * confirmation

        return signal.astype(float)
