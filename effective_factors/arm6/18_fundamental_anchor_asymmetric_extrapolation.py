"""Fundamental-anchor extrapolation penalty.

Mechanism: industry-relative price appreciation is more likely to be an
extrapolative overvaluation when a firm lacks value, profitability, and
conservative-investment support.  The signal retains a slow fundamental anchor
but applies a price-run-up penalty only to its weak-anchor tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


def _cs_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Defensive cross-sectional percentile rank with neutral missing values."""
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return clean.rank(axis=1, pct=True).fillna(0.5)


def _industry_relative_return(
    close: pd.DataFrame, industry: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Causal equal-weighted industry-relative cumulative return.

    Industry labels are static in this feed.  The first row is therefore used
    only as a cross-sectional identifier, while every price observation is at
    or before the current date.
    """
    raw_return = close.pct_change(horizon)
    labels = industry.iloc[0].where(industry.iloc[0].notna(), "__ungrouped__").astype(str)
    relative = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    for label in pd.unique(labels):
        members = labels.index[labels == label]
        group_return = raw_return.loc[:, members].mean(axis=1)
        relative.loc[:, members] = raw_return.loc[:, members].sub(group_return, axis=0)

    return relative


@register_factor
class FundamentalAnchorAsymmetricExtrapolation(BaseFactor):
    """Condition slow fundamental anchors on asymmetric relative-price chasing."""

    factor_id = "fundamental_anchor_asymmetric_extrapolation"
    name = "Fundamental-Anchor Extrapolation Penalty"
    category = "statistical_arbitrage"
    description = (
        "Fundamental value-quality-investment rank minus a penalty for positive "
        "industry-relative price run-ups, with the penalty concentrated in the "
        "weak-fundamental tail where extrapolation is least well anchored."
    )
    window_length = 20
    inputs = [
        "close",
        "industry",
        "assetGrowth",
        "roic",
        "earningsYield",
        "freeCashFlowYield",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].replace([np.inf, -np.inf], np.nan)

        # A deliberately simple, slow-moving successor to the parent's three
        # sleeves: cheap, profitable, and conservatively investing firms.
        anchor = (
            _cs_rank(data["earningsYield"])
            + _cs_rank(data["freeCashFlowYield"])
            + _cs_rank(data["roic"])
            + _cs_rank(-data["assetGrowth"])
        ) / 4.0

        relative_return = _industry_relative_return(close, data["industry"], 15)
        daily_return = close.pct_change()
        volatility = daily_return.rolling(20, min_periods=20).std()
        standardized_runup = relative_return.div(
            volatility.mul(np.sqrt(15.0)).replace(0.0, np.nan)
        )

        # A negative relative move is not treated as evidence of extrapolation.
        # Among positive moves, percentile ranking makes the signal robust to
        # changing cross-sectional volatility and outlier price jumps.
        positive_runup = standardized_runup.where(standardized_runup > 0.0)
        runup_penalty = positive_runup.rank(axis=1, pct=True).fillna(0.0)

        # Strong anchors are not mechanically faded; weak anchors lose score
        # only when their recent relative-price path invites extrapolation.
        score = anchor - runup_penalty * (1.0 - anchor)
        return score.rank(axis=1, pct=True).fillna(0.5).astype(float)
