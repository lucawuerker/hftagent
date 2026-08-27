from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.factors.base import BaseFactor
from quant_fund_agent.factors.ops import rank
from quant_fund_agent.factors.registry import register_factor


def _group_mean(x: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional group mean using static memberships."""
    labels = labels_df.ffill().iloc[-1].fillna("Unknown").astype(str)
    return x.T.groupby(labels, sort=False).transform("mean").T


def _leave_one_out_mean(
    x: pd.DataFrame, labels_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Leave-one-out group mean and static group membership count."""
    labels = labels_df.ffill().iloc[-1].fillna("Unknown").astype(str)
    counts = labels.map(labels.value_counts()).astype(float)
    group_mean = x.T.groupby(labels, sort=False).transform("mean").T
    peer_mean = (
        group_mean.mul(counts, axis=1) - x
    ).div((counts - 1.0).replace(0.0, np.nan), axis=1)
    return peer_mean, counts


@register_factor
class IndustryRevisionAdoptionGapV2(BaseFactor):
    factor_id = "industry_revision_adoption_gap_v2"
    name = "Industry Revision Adoption Gap"
    category = "sentiment"
    description = (
        "Long firms with unassimilated, persistent leave-one-out industry "
        "estimate revisions when their own consensus is stale and their "
        "industry-adjusted price response remains muted."
    )
    window_length = 60
    inputs = [
        "close", "epsEstimate", "revenueEstimate", "industry", "sector"
    ]
    prediction_horizon = 6
    suggested_horizons = [6]

    def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        close = data["close"].astype(float)
        eps = data["epsEstimate"].reindex_like(close).astype(float)
        revenue = data["revenueEstimate"].reindex_like(close).astype(float)
        industry = data["industry"].reindex_like(close)
        sector = data["sector"].reindex_like(close)

        eps_lag = eps.shift(1)
        revenue_lag = revenue.shift(1)
        eps_changed = eps.notna() & eps_lag.notna() & eps.ne(eps_lag)
        revenue_changed = (
            revenue.notna() & revenue_lag.notna() & revenue.ne(revenue_lag)
        )

        eps_revision = (
            (eps - eps_lag) / (eps_lag.abs() + 0.25)
        ).clip(-2.0, 2.0)
        revenue_revision = (
            (revenue - revenue_lag) / (revenue_lag.abs() + 1000000.0)
        ).clip(-2.0, 2.0)
        revision = (
            0.8 * eps_revision.where(eps_changed, 0.0).fillna(0.0)
            + 0.2 * revenue_revision.where(revenue_changed, 0.0).fillna(0.0)
        )

        peer_revision, industry_count = _leave_one_out_mean(revision, industry)
        peer_state = peer_revision.ewm(
            halflife=10.0, adjust=False, min_periods=1
        ).mean()

        own_update = (eps_changed | revenue_changed).astype(float)
        own_activity = own_update.ewm(
            halflife=10.0, adjust=False, min_periods=1
        ).mean()
        consensus_staleness = 1.0 - own_activity.clip(0.0, 1.0)

        ret = close.replace(0.0, np.nan).pct_change(fill_method=None)
        peer_ret, _ = _leave_one_out_mean(ret, industry)
        idio_ret = (ret - peer_ret).clip(-0.25, 0.25)
        idio_move = idio_ret.rolling(10, min_periods=6).sum()
        idio_vol = idio_ret.rolling(60, min_periods=20).std()
        response_z = idio_move / (idio_vol * np.sqrt(10.0)).replace(0.0, np.nan)
        muted_response = np.exp(-response_z.abs().clip(0.0, 3.0)).fillna(0.0)

        raw_signal = peer_state * consensus_staleness * muted_response
        sector_adjusted = raw_signal - _group_mean(raw_signal, sector)

        valid_consensus = eps.notna() | revenue.notna()
        eligible = valid_consensus & industry_count.gt(1)
        signal = sector_adjusted.where(eligible)
        return rank(signal).fillna(0.5)
