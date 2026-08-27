"""Salient accrual-growth extrapolation.

Mechanism: market participants extrapolate visible revenue-growth narratives more
strongly than they process the working-capital and cash-flow details that qualify
those narratives.  A fresh, high-attention revenue update is therefore vulnerable
when it coincides with expensive valuation, weak operating-cash-flow growth,
rising receivables, and a worsening cash-conversion cycle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.registry import register_factor


def _groups(industry: pd.DataFrame, columns: pd.Index) -> pd.Series:
    """Static industry labels, with singleton fallbacks for missing labels."""
    labels = industry.ffill().iloc[0].reindex(columns).astype(object)
    fallback = pd.Series(
        ["__singleton_" + str(column) for column in columns],
        index=columns,
        dtype=object,
    )
    return labels.where(labels.notna(), fallback)


def _industry_percentile(values: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    clean = values.replace([np.inf, -np.inf], np.nan)
    return clean.T.groupby(groups, sort=False).rank(pct=True).T


def _industry_demean(values: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    clean = values.replace([np.inf, -np.inf], np.nan)
    means = clean.T.groupby(groups, sort=False).transform("mean").T
    return clean - means


def _event_mask(values: pd.DataFrame) -> pd.DataFrame:
    """True at the first available observation and at later PIT updates."""
    prior = values.shift(1)
    return values.notna() & (prior.isna() | values.ne(prior))


def _event_age(events: pd.DataFrame) -> pd.DataFrame:
    """Causal bars elapsed since the latest event for every ticker."""
    event_array = events.fillna(False).to_numpy(dtype=bool)
    n_rows, n_cols = event_array.shape
    positions = np.arange(n_rows, dtype=float)[:, None]
    last_event = np.maximum.accumulate(
        np.where(event_array, positions, -1.0), axis=0
    )
    age = positions - last_event
    age[last_event < 0.0] = np.nan
    return pd.DataFrame(age, index=events.index, columns=events.columns)


def _mean_available(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Average panel components without treating unavailable accounting fields as zero."""
    total = sum(part.fillna(0.0) for part in parts)
    count = sum(part.notna().astype(float) for part in parts)
    return total / count.replace(0.0, np.nan)


@register_factor
class SalientAccrualGrowthExtrapolationShort(BaseFactor):
    factor_id = "salient_accrual_growth_extrapolation_short"
    name = "Salient Accrual Growth Extrapolation"
    category = "statistical_arbitrage"
    description = (
        "Shorts fresh, heavily attended, expensive revenue-growth narratives when "
        "cash-flow growth, receivables, and cash-conversion-cycle data contradict "
        "the apparent operating momentum."
    )
    window_length = 30
    inputs = [
        "close",
        "volume",
        "industry",
        "revenueGrowth",
        "operatingCashFlowGrowth",
        "receivablesGrowth",
        "cashConversionCycle",
        "psRatio",
        "freeCashFlowYield",
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].apply(pd.to_numeric, errors="coerce")
        volume = data["volume"].apply(pd.to_numeric, errors="coerce")
        groups = _groups(data["industry"], close.columns)

        revenue_growth = data["revenueGrowth"].apply(pd.to_numeric, errors="coerce")
        cashflow_growth = data["operatingCashFlowGrowth"].apply(
            pd.to_numeric, errors="coerce"
        )
        receivables_growth = data["receivablesGrowth"].apply(
            pd.to_numeric, errors="coerce"
        )
        cash_cycle = data["cashConversionCycle"].apply(pd.to_numeric, errors="coerce")
        cycle_change = cash_cycle - cash_cycle.shift(1)

        sales_multiple = data["psRatio"].apply(pd.to_numeric, errors="coerce")
        fcf_yield = data["freeCashFlowYield"].apply(pd.to_numeric, errors="coerce")

        revenue_rank = _industry_percentile(revenue_growth.clip(-3.0, 3.0), groups)
        weak_cash_rank = 1.0 - _industry_percentile(
            cashflow_growth.clip(-3.0, 3.0), groups
        )
        receivable_rank = _industry_percentile(
            receivables_growth.clip(-3.0, 3.0), groups
        )
        cycle_rank = _industry_percentile(cycle_change.clip(-365.0, 365.0), groups)

        # A high value identifies sales growth that is increasingly financed by
        # working capital rather than translated into operating cash generation.
        accrual_narrative_risk = _mean_available(
            [revenue_rank, weak_cash_rank, receivable_rank, cycle_rank]
        )

        expensive_sales = _industry_percentile(
            sales_multiple.where(sales_multiple > 0.0).clip(upper=100.0), groups
        )
        expensive_cashflow = 1.0 - _industry_percentile(
            fcf_yield.clip(-2.0, 2.0), groups
        )
        valuation_salience = _mean_available([expensive_sales, expensive_cashflow])

        event = (
            _event_mask(revenue_growth)
            | _event_mask(cashflow_growth)
            | _event_mask(receivables_growth)
            | _event_mask(cash_cycle)
        )
        age = _event_age(event)
        freshness = np.exp(-np.log(2.0) * age / 5.0).where(age <= 15.0, 0.0)

        volume_base = volume.rolling(20, min_periods=8).mean().replace(0.0, np.nan)
        relative_turnover = (volume / volume_base).clip(0.0, 6.0)
        attention = ((relative_turnover - 0.8) / 1.2).clip(0.0, 1.5)

        overextrapolation = (
            accrual_narrative_risk * valuation_salience * freshness * attention
        )
        signal = -_industry_demean(overextrapolation, groups)

        return (
            signal.reindex(index=close.index, columns=close.columns)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
        )
