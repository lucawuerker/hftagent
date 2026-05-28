"""Strategy-level correlation & covariance estimation.

The PM agent needs a covariance matrix between every strategy in its
universe.  We build it from each strategy's persisted per-bar portfolio
return series (saved by ``StrategyDatabase.register_strategy``).

Strategies may be backtested over slightly different windows
(researcher sessions, OOS windows, etc.).  We therefore align on the
intersection of their indices before computing correlations, and warn
if the overlap is too short.
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("portfolio.correlations")

MIN_OVERLAP_BARS = 30


def compute_correlation_matrix(
    strategy_returns: Mapping[str, pd.Series],
    min_overlap: int = MIN_OVERLAP_BARS,
) -> pd.DataFrame:
    """Pairwise Pearson correlation between strategy return series.

    Strategies are aligned on the *union* of their indices (NaN-filled
    where missing); ``df.corr(min_periods=...)`` then handles overlap.
    Pairs with fewer than ``min_overlap`` overlapping bars are returned
    as NaN.

    Returns an empty DataFrame if fewer than 2 strategies are passed.
    """
    if len(strategy_returns) < 2:
        log.debug("compute_correlation_matrix: <2 strategies, returning empty.")
        return pd.DataFrame()

    aligned = pd.DataFrame(
        {sid: s.astype(float) for sid, s in strategy_returns.items()},
    )
    corr = aligned.corr(method="pearson", min_periods=min_overlap)
    # Symmetrise + reset diagonal to exactly 1.  We copy to numpy because
    # the DataFrame ``corr`` returns can be backed by a read-only ndarray.
    arr = np.array(corr.values, copy=True)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def compute_covariance_matrix(
    strategy_returns: Mapping[str, pd.Series],
    min_overlap: int = MIN_OVERLAP_BARS,
    annualisation_factor: float = 1.0,
) -> pd.DataFrame:
    """Empirical covariance matrix from per-bar returns.

    Pass ``annualisation_factor = bars_per_year`` to get annualised
    covariance.  Default is 1.0 (covariance in the same units as the
    input returns).
    """
    if len(strategy_returns) < 2:
        # Single strategy: just its variance.
        if len(strategy_returns) == 1:
            sid, s = next(iter(strategy_returns.items()))
            return pd.DataFrame([[float(s.var(ddof=1) * annualisation_factor)]],
                                index=[sid], columns=[sid])
        return pd.DataFrame()

    aligned = pd.DataFrame({sid: s.astype(float) for sid, s in strategy_returns.items()})
    cov = aligned.cov(min_periods=min_overlap) * annualisation_factor
    arr = np.array(cov.values, copy=True)
    arr = (arr + arr.T) / 2.0
    return pd.DataFrame(arr, index=cov.index, columns=cov.columns)


def covariance_from_vols_and_corr(
    vols: Mapping[str, float],
    correlation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct a covariance matrix from per-strategy volatilities + ρ.

    Useful when full return series are not available (e.g. only the
    summary stats in StrategyRecord.backtest_metrics are present) but we
    still have an estimate of pairwise correlations.

        Σ_ij = ρ_ij · σ_i · σ_j
    """
    sids = list(vols.keys())
    sigma = np.array([float(vols[s]) for s in sids], dtype=float)
    n = len(sids)

    if n == 0:
        return pd.DataFrame()
    if correlation_matrix.empty:
        return pd.DataFrame(np.diag(sigma ** 2), index=sids, columns=sids)

    rho = np.eye(n)
    for i, si in enumerate(sids):
        for j, sj in enumerate(sids):
            if i == j:
                continue
            r = correlation_matrix.get(sj, {}).get(si) if hasattr(correlation_matrix, "get") else None
            if r is None or pd.isna(r):
                try:
                    r = float(correlation_matrix.loc[si, sj])
                except (KeyError, TypeError):
                    r = 0.0
                if pd.isna(r):
                    r = 0.0
            rho[i, j] = float(r)
    cov = (sigma[:, None] * sigma[None, :]) * rho
    return pd.DataFrame(cov, index=sids, columns=sids)
