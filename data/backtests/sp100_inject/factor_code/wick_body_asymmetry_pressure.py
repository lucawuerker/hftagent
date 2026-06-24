"""Large intrabar wicks relative to the candle body usually indicate failed price discovery: one side of the move was aggressively rejected before the bar closed. When the upper wick dominates, buyers likely overextended and were faded; when the lower wick dominates, sellers were rejected and price often snaps back. This is a structural price-action version of the rejection logic emphasized by the sensor-fusion paper’s focus on distinguishing subtle movement asymmetries and merging complementary evidence rather than trusting a single noisy cue."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


@register_factor
class WickBodyAsymmetryPressure(BaseFactor):
    """Signed rejection pressure from wick/body asymmetry normalized by range."""

    factor_id = "wick_body_asymmetry_pressure"
    name = "Wick-Body Asymmetry Pressure"
    category = "mean_reversion"
    description = (
        "Compute a signed rejection score from each bar’s upper and lower wicks "
        "normalized by true range and body size. Positive values indicate stronger "
        "downside rejection (bullish mean reversion), while negative values indicate "
        "stronger upside rejection (bearish mean reversion); the output is cross-sectional "
        "over tickers for each timestamp."
    )
    window_length = 1
    inputs = ["open", "high", "low", "close"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        open_ = data["open"].astype(float)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        body = (close - open_).abs()
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)
        true_range = (high - low).abs()

        eps = 1e-12
        norm = true_range + body + eps
        asymmetry = (lower_wick - upper_wick) / norm

        wick_intensity = (upper_wick + lower_wick) / norm
        body_sign = np.sign(close - open_)

        raw = asymmetry * (1.0 + wick_intensity) - 0.25 * body_sign * (body / norm)

        # Cross-sectional normalization per timestamp to express relative pressure.
        centered = raw.sub(raw.mean(axis=1), axis=0)
        scale = centered.abs().sum(axis=1).replace(0.0, np.nan)
        out = centered.div(scale, axis=0).fillna(0.0)

        return out
