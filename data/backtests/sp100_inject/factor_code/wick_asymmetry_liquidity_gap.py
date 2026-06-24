"""Large upper or lower wicks indicate that price explored a level but was rejected before the bar close, which is often a footprint of transient liquidity withdrawal or aggressive absorption. In the order-book view of Lachapelle et al. (2013), such failed excursions are consistent with temporary imbalances that do not fully persist once liquidity providers re-enter; the signal should therefore predict short-horizon continuation only when the wick is one-sided and the close remains near the opposite end. Cross-sectionally, names with strong asymmetry between upper and lower rejection should exhibit a more informative local imbalance than plain range or return features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, rank, ts_mean
from quant_fund_agent.factors.registry import register_factor


@register_factor
class WickAsymmetryLiquidityGap(BaseFactor):
    """One-sided wick asymmetry normalized by range and liquidity."""

    factor_id = "wick_asymmetry_liquidity_gap"
    name = "Wick Asymmetry Liquidity Gap"
    category = "microstructure"
    description = (
        "Measures the difference between upper and lower wick size, normalized by total "
        "intrabar range and volume, so that large one-sided rejection stands out while "
        "symmetric noise cancels. Positive values indicate dominant lower-wick rejection "
        "(bullish absorption), negative values indicate dominant upper-wick rejection "
        "(bearish absorption)."
    )
    window_length = 10
    inputs = ["open", "high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
        range_ = (high - low).replace(0.0, np.nan)

        wick_asym = (lower_wick - upper_wick) / range_
        body_pos = ((close - open_) / range_).clip(lower=-1.0, upper=1.0)

        volume_rel = volume / adv(volume.fillna(0.0), 20).replace(0.0, np.nan)
        liquidity_penalty = ts_mean(volume_rel.fillna(0.0), 3)

        raw = wick_asym * (1.0 - body_pos.abs())
        raw = raw * (1.0 / (1.0 + liquidity_penalty.fillna(0.0)))

        cs = rank(raw.fillna(0.0))
        return (raw.fillna(0.0) + 0.25 * (cs - 0.5)).astype(float)
