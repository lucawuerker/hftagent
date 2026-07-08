"""Shared standardisation helpers for the comparison tracks.

The **per-underlying (time-series) z-score** is the non-cross-sectional
standardiser used across the brute-force and analytics tracks whenever
``ComparisonConfig.fit_standardize == "per_underlying"`` (the default): each
factor signal is standardised *per column over its own history* — optionally
using only the in-sample window's statistics so there is no out-of-sample leakage
— rather than cross-sectionally across tickers at each timestamp.

Centralising it here guarantees the model-facing tracks can't drift, and makes
them work with a *single* underlying (a cross-sectional z-score over one name is
all-NaN; a per-underlying time-series z-score is well defined).  Raw factor IC is
computed per underlying and therefore does not need this standardisation step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_underlying_zscore(sig: pd.DataFrame, stat_idx=None) -> pd.DataFrame:
    """Z-score each column over time (mean 0, std 1 per underlying).

    ``±inf`` is treated as missing first (otherwise it poisons the mean/std and
    leaks into any downstream fit).  ``stat_idx`` (a row index/mask) restricts the
    mean/std *statistics* to those rows — pass the in-sample window to avoid
    out-of-sample leakage — while the transform is applied to every row; ``None``
    uses the whole frame.  A constant column (std 0) or one with no data yields
    NaN for that column (left for the caller to impute), never a divide error.
    """
    s = sig.replace([np.inf, -np.inf], np.nan)
    ref = s.loc[stat_idx] if stat_idx is not None else s
    mu = ref.mean()
    sd = ref.std().replace(0.0, np.nan)
    return (s - mu) / sd
