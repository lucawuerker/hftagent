"""The paper on institutional price impact asymmetry suggests that sells often move prices more than buys when liquidity is thinner and the market is under stress. Using only OHLCV, we can proxy this imbalance by comparing how much of the bar’s total range is achieved on the downside versus the upside, conditioned on abnormal volume. Bars with heavy volume and a close near the low imply that sell-side pressure is consuming liquidity more aggressively than buy-side demand, which should predict short-term continuation in that direction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import adv, ts_mean, ts_rank
from quant_fund_agent.factors.registry import register_factor


@register_factor
class RangeImbalanceImpactAsymmetry(BaseFactor):
    """Range-based sell/buy impact asymmetry conditioned on abnormal volume."""

    factor_id = "range_imbalance_impact_asymmetry"
    name = "Range Imbalance Impact Asymmetry"
    category = "microstructure"
    description = (
        "Compute a signed imbalance from the intrabar position of the close "
        "within the high-low range, then amplify it by volume relative to a "
        "rolling baseline and by range expansion. Positive values indicate "
        "high-volume downside dominance; negative values indicate high-volume "
        "upside dominance."
    )
    window_length = 20
    inputs = ["high", "low", "close", "volume"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        rng = (high - low).replace(0, np.nan)
        close_pos = ((close - low) / rng).clip(lower=0.0, upper=1.0)

        # Signed intrabar imbalance: positive when close is near the low,
        # negative when close is near the high.
        imbalance = (0.5 - close_pos) * 2.0

        # Abnormal volume relative to its rolling baseline.
        vol_base = ts_mean(volume, 20)
        vol_rel = (volume / vol_base.replace(0, np.nan)) - 1.0
        vol_rel = vol_rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol_weight = ts_rank(vol_rel.abs(), 20).fillna(0.0)

        # Range expansion versus recent average range.
        range_base = ts_mean(rng, 20)
        range_expansion = (rng / range_base.replace(0, np.nan)) - 1.0
        range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        range_weight = ts_rank(range_expansion.clip(lower=0.0), 20).fillna(0.0)

        # Prefer heavy-volume bars and expanded ranges, with signed direction.
        intensity = (1.0 + vol_rel.clip(lower=0.0)) * (1.0 + range_expansion.clip(lower=0.0))
        intensity = intensity.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        asymmetry = imbalance * intensity * (0.5 + 0.5 * vol_weight) * (0.5 + 0.5 * range_weight)
        asymmetry = asymmetry.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return asymmetry.reindex(index=close.index, columns=close.columns)
