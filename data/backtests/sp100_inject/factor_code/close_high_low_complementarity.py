"""Close High-Low Complementarity: When the bar closes near one extreme while the opposite extreme remains “untouched,” price discovery is often incomplete: the market has probed both directions but only one side has been accepted by the close. This factor treats that asymmetry as a complementary-information signal, analogous to the paper’s idea that richer representations reduce uncertainty; here, a close that is well-supported by a strong intrabar excursion but contradicts the bar’s directional range anatomy tends to be less reliable than a close that is confirmed by a balanced path. In practice, strong complementary range structure often precedes short-horizon continuation when the close is near the high after a shallow lower wick, and reversion when the close is near the low after a shallow upper wick, especially after elevated intrabar volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor
from quant_fund_agent.factors.ops import rank, ts_mean, ts_rank, abs_, sign


@register_factor
class CloseHighLowComplementarity(BaseFactor):
    """Complementarity of close location and wick asymmetry within the bar."""

    factor_id = "close_high_low_complementarity"
    name = "Close High-Low Complementarity"
    category = "other"
    description = (
        "Compute a cross-sectional score from the interaction of close location within the high-low range "
        "and the relative size of the upper/lower wicks, scaled by the bar’s true range. Positive values "
        "indicate closes near the high with weak upper rejection and strong lower excursion; negative values "
        "indicate closes near the low with weak lower rejection and strong upper excursion."
    )
    window_length = 20
    inputs = ["high", "low", "close", "open"]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        open_ = data["open"]

        eps = 1e-12
        rng = (high - low).abs()
        safe_rng = rng.replace(0, np.nan)

        close_loc = ((close - low) / safe_rng).clip(0.0, 1.0).fillna(0.5)
        upper_wick = (high - np.maximum(open_, close)).clip(lower=0.0)
        lower_wick = (np.minimum(open_, close) - low).clip(lower=0.0)

        wick_total = (upper_wick + lower_wick).replace(0, np.nan)
        wick_imbalance = ((lower_wick - upper_wick) / wick_total).fillna(0.0)

        body = (close - open_).abs()
        body_share = (body / (safe_rng + eps)).fillna(0.0)

        proximity = 2.0 * close_loc - 1.0
        complementarity = proximity * wick_imbalance

        intrabar_support = (lower_wick - upper_wick) / (safe_rng + eps)
        intrabar_support = intrabar_support.fillna(0.0)

        tr_scaled = ts_rank(ts_mean(rng, 5), 20)
        magnitude = rank(abs_(complementarity) + 0.5 * abs_(intrabar_support) + 0.25 * body_share)

        raw = (complementarity + 0.5 * intrabar_support) * (1.0 + tr_scaled) * magnitude
        return raw.fillna(0.0) * sign(close - open_)
