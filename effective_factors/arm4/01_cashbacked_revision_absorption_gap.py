"""Analyst estimate changes are more likely to represent durable information when the issuer has historically converted reported earnings into operating cash flow. Investors frequently react to the headline revision while underweighting the cash-validation distinction, particularly when the price response is initially muted or the update arrives amid many concurrent revisions. Go long cash-backed positive joint EPS/revenue revisions with below-peer price absorption and short the symmetric negative cases; the hypothesis fails if cash quality does not improve the subsequent return spread after controlling for revision size and industry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import indneutralize, rank
from quant_fund_agent.factors.registry import register_factor


def _robust_cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    """Causal row-wise robust z-score using the contemporaneous universe only."""
    median = values.median(axis=1)
    centered = values.sub(median, axis=0)
    mad_scale = 1.4826 * centered.abs().median(axis=1)
    std_scale = values.std(axis=1)
    scale = mad_scale.where(mad_scale > 1.0e-12, std_scale)
    scale = scale.where(scale > 1.0e-12)
    return centered.div(scale, axis=0).clip(-5.0, 5.0)


def _fractional_revision(estimate: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Estimate revision relative to the latest prior available consensus."""
    prior = estimate.shift(lag)
    denominator = prior.abs().where(prior.abs() > 1.0e-10)
    return (estimate - prior) / denominator


def _cash_confirmation(
    income_quality: pd.DataFrame,
    operating_cash_flow: pd.DataFrame,
    net_income: pd.DataFrame,
) -> pd.DataFrame:
    """Bounded cash-conversion confirmation, with no support when unavailable."""
    income_scale = net_income.abs().where(net_income.abs() > 1.0e-10)
    cash_to_earnings = (operating_cash_flow / income_scale).clip(-5.0, 5.0)
    reported_quality = income_quality.clip(-5.0, 5.0)

    quality = 0.5 * (reported_quality + cash_to_earnings)
    confirmation = 0.5 + 0.5 * np.tanh(quality - 1.0)
    available = reported_quality.notna() | cash_to_earnings.notna()
    return confirmation.where(available, 0.0).fillna(0.0)


@register_factor
class CashbackedRevisionAbsorptionGap(BaseFactor):
    """Cash-confirmed joint estimate revisions not yet reflected in price."""

    factor_id = "cashbacked_revision_absorption_gap"
    name = "Cash-Backed Revision Absorption Gap"
    category = "sentiment"
    description = (
        "Ranks industry-neutral gaps between cash-confirmed, concordant recent "
        "EPS and revenue estimate revisions and the stock's own recent price "
        "response, favoring durable revisions that remain under-absorbed."
    )
    window_length = 42
    inputs = [
        "close",
        "epsEstimate",
        "revenueEstimate",
        "incomeQuality",
        "operatingCashFlow",
        "netIncome",
        "industry",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"]
        eps_estimate = data["epsEstimate"]
        revenue_estimate = data["revenueEstimate"]

        # Quarterly point-in-time estimates are deliberately compared with a
        # slow positive lag: this captures a still-recent revision without
        # manufacturing changes between forward-filled report dates.
        eps_revision = _fractional_revision(eps_estimate, 21)
        revenue_revision = _fractional_revision(revenue_estimate, 21)
        eps_z = _robust_cross_sectional_zscore(eps_revision)
        revenue_z = _robust_cross_sectional_zscore(revenue_revision)

        # The concordance gate suppresses one-dimensional or conflicting
        # revisions while retaining the sign of aligned positive/negative news.
        agreement = 0.5 + 0.5 * np.tanh(eps_z * revenue_z)
        joint_revision = 0.5 * (eps_z + revenue_z) * agreement

        cash_backing = _cash_confirmation(
            data["incomeQuality"],
            data["operatingCashFlow"],
            data["netIncome"],
        )
        validated_revision = joint_revision.fillna(0.0) * cash_backing

        # Price absorption is measured over the same trailing interval and
        # standardized against the contemporaneous investable cross-section.
        price_response = close / close.shift(21) - 1.0
        response_z = _robust_cross_sectional_zscore(price_response).fillna(0.0)
        absorption_gap = validated_revision - response_z

        industry = data["industry"].fillna("Unknown")
        peer_adjusted_gap = indneutralize(absorption_gap.fillna(0.0), industry)
        return rank(peer_adjusted_gap).fillna(0.5)
