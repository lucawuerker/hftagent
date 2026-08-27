"""Industry-residual estimate persistence.

Mechanism: broad industry estimate changes are often quickly impounded common news,
while coherent estimate innovations that are exceptional relative to direct peers
contain issuer-specific information that diffuses more slowly through the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


def _scaled_update(values: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Causal robust estimate-update impulse scaled by prior nonzero updates."""
    change = values.diff(1)
    prior_changes = change.abs().where(change.ne(0.0))
    scale = prior_changes.rolling(lookback, min_periods=2).median()
    valid = values.notna() & values.shift(1).notna()
    return (change / scale.replace(0.0, np.nan)).where(valid).clip(-4.0, 4.0)


def _decayed_coherent_revision(
    eps_update: pd.DataFrame,
    revenue_update: pd.DataFrame,
    halflife: float,
) -> pd.DataFrame:
    """Retain only same-direction, dual-channel estimate information causally."""
    same_sign = (
        eps_update.notna()
        & revenue_update.notna()
        & eps_update.ne(0.0)
        & revenue_update.ne(0.0)
        & np.sign(eps_update).eq(np.sign(revenue_update))
    )
    joint = (0.5 * (eps_update + revenue_update)).where(same_sign, 0.0)
    observed = same_sign.astype(float)
    decayed_signal = joint.ewm(halflife=halflife, adjust=False, min_periods=1).mean()
    decayed_observed = observed.ewm(halflife=halflife, adjust=False, min_periods=1).mean()
    return decayed_signal.where(decayed_observed.gt(0.02))


@register_factor
class IndustryResidualEstimatePersistence(BaseFactor):
    factor_id = "industry_residual_estimate_persistence"
    name = "Industry-Residual Estimate Persistence"
    category = "sentiment"
    description = (
        "Decayed coherent EPS and revenue estimate innovations, with the common "
        "industry revision component removed to isolate firm-specific information drift."
    )
    window_length = 252
    inputs = ["close", "epsEstimate", "revenueEstimate", "industry"]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        eps_update = _scaled_update(data["epsEstimate"], 252)
        revenue_update = _scaled_update(data["revenueEstimate"], 252)

        persistent_revision = _decayed_coherent_revision(
            eps_update, revenue_update, 6.0
        )
        industry_residual = indneutralize(persistent_revision, data["industry"])

        return (rank(industry_residual) - 0.5).reindex_like(close).fillna(0.0)
