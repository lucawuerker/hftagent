"""Cash-validated opening-auction rejection continuation.

Mechanism: a rejected opening gap can reflect temporary auction liquidity pressure
rather than a repudiation of information.  When a fresh estimate revision agrees
with cash-flow validation, the gap direction is more likely informed institutional
demand that was absorbed intraday and subsequently continues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _revision_impulse(estimate: pd.DataFrame, span: int) -> pd.DataFrame:
    """Causal, bounded percentage estimate revision with exponential persistence."""
    previous = estimate.shift(1)
    denominator = previous.abs().clip(lower=1.0e-8)
    change = ((estimate - previous) / denominator).clip(lower=-1.0, upper=1.0)
    change = change.where(estimate.notna() & previous.notna(), 0.0)
    return change.ewm(span=span, adjust=False, min_periods=1).mean()


def _auction_gap_continuation(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Signed gap direction weighted by intraday rejection and abnormality."""
    prior_close = close.shift(1)
    valid = (open_ > 0.0) & (prior_close > 0.0)
    gap = np.log(open_.where(valid) / prior_close.where(valid))
    scale = gap.abs().rolling(20, min_periods=10).median().replace(0.0, np.nan)
    standardized = gap.div(scale).clip(lower=-6.0, upper=6.0)

    bar_high = pd.DataFrame(
        np.maximum(high.to_numpy(), close.to_numpy()),
        index=close.index,
        columns=close.columns,
    )
    bar_low = pd.DataFrame(
        np.minimum(low.to_numpy(), close.to_numpy()),
        index=close.index,
        columns=close.columns,
    )
    bar_range = (bar_high - bar_low).replace(0.0, np.nan)
    close_location = ((2.0 * close - bar_high - bar_low) / bar_range).clip(-1.0, 1.0)
    gap_sign = np.sign(standardized)
    rejection = ((1.0 - gap_sign * close_location) / 2.0).clip(0.0, 1.0)
    abnormality = standardized.abs().clip(upper=3.0) / 3.0
    return standardized * rejection * abnormality


@register_factor
class CashValidatedAuctionRejectionContinuation(BaseFactor):
    factor_id = 'cash_validated_auction_rejection_continuation'
    name = 'Cash-Validated Auction Continuation'
    category = 'microstructure'
    description = (
        'Gap-direction continuation after intraday opening-auction rejection, '
        'activated only by fresh, cash-flow-confirmed analyst estimate revisions.'
    )
    window_length = 63
    inputs = [
        'open',
        'high',
        'low',
        'close',
        'epsEstimate',
        'revenueEstimate',
        'operatingCashFlowGrowth',
        'freeCashFlowGrowth',
        'incomeQuality',
        'changeInWorkingCapital',
        'revenue',
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data['close']
        eps_revision = _revision_impulse(data['epsEstimate'], 12)
        revenue_revision = _revision_impulse(data['revenueEstimate'], 12)
        revision = 0.6 * eps_revision.fillna(0.0) + 0.4 * revenue_revision.fillna(0.0)
        revision_coverage = (
            data['epsEstimate'].notna().astype(float)
            + data['revenueEstimate'].notna().astype(float)
        ) / 2.0

        revenue = data['revenue'].abs().replace(0.0, np.nan)
        working_capital_intensity = data['changeInWorkingCapital'].div(revenue)
        cash_validation = (
            (2.0 * rank(data['operatingCashFlowGrowth']) - 1.0).fillna(0.0)
            + (2.0 * rank(data['freeCashFlowGrowth']) - 1.0).fillna(0.0)
            + (2.0 * rank(data['incomeQuality']) - 1.0).fillna(0.0)
            + (1.0 - 2.0 * rank(working_capital_intensity)).fillna(0.0)
        ) / 4.0
        cash_coverage = (
            data['operatingCashFlowGrowth'].notna().astype(float)
            + data['freeCashFlowGrowth'].notna().astype(float)
            + data['incomeQuality'].notna().astype(float)
            + working_capital_intensity.notna().astype(float)
        ) / 4.0

        revision_strength = rank(revision.abs()).fillna(0.0)
        directional_agreement = (np.sign(revision) * cash_validation).clip(lower=0.0, upper=1.0)
        credibility = revision_strength * directional_agreement * revision_coverage * cash_coverage

        auction_continuation = _auction_gap_continuation(
            data['open'], data['high'], data['low'], close
        )
        signal = auction_continuation * credibility
        return rank(signal).reindex_like(close).fillna(0.5)
