"""Turn the factor-signal panel into a supervised learning dataset.

The Architect's models are cross-sectional return predictors: at each bar we
use the (normalised) factor signals for every ticker as features ``X`` and the
ticker's *forward* return as the target ``y``.  This module flattens the
``{factor_id: (time × ticker) DataFrame}`` panel into a long
``(samples × features)`` matrix and unflattens predictions back into a
``(time × ticker)`` signal frame.

No-lookahead guarantee
----------------------
The target is ``forward_returns(close, horizon)`` — i.e. the return realised
*after* the bar whose features we use, so a row at time ``t`` is never trained
against information available only at ``t``.  In the backtester the position
formed from a signal at ``t`` is additionally applied with a one-bar delay, so
there is no same-bar leakage either.

Normalisation contract
-----------------------
Features are expected to be the **already cross-sectionally normalised** factor
signals (the backtester z-scores signals before calling ``strategy.calc``).
Training therefore normalises once up front and prediction receives the
already-normalised signals — the two paths build identical features.  Missing
(NaN) features are imputed to ``0`` (the neutral value post-z-score) in both
paths rather than dropped, so a sparse factor cannot collapse the sample set.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns

# Fitting cross-sectional return predictors on the full intraday panel can mean
# millions of (timestamp, ticker) rows.  That gives the CV linear models
# (``LassoCV`` / ``ElasticNetCV``, coordinate descent over an alpha grid × folds)
# no statistical benefit but dominates wall-clock — a single fit can take many
# minutes.  We therefore cap the rows actually handed to ``estimator.fit`` via a
# deterministic uniform subsample.  ``MODELING_MAX_TRAIN_SAMPLES=0`` disables it.
DEFAULT_MAX_TRAIN_SAMPLES = 200_000


def _max_train_samples() -> int | None:
    raw = os.getenv("MODELING_MAX_TRAIN_SAMPLES")
    if raw is None or raw == "":
        return DEFAULT_MAX_TRAIN_SAMPLES
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_TRAIN_SAMPLES
    return n if n > 0 else None


def _stack(df: pd.DataFrame) -> pd.Series:
    """Stack ``(time × ticker)`` → Series indexed by ``(timestamp, ticker)``.

    Keeps NaNs (we drop them later, jointly across features and target) and is
    robust to the pandas 2.x ``stack`` signature change.
    """
    try:
        return df.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        return df.stack(dropna=False)


def stack_features(
    factor_signals: dict[str, pd.DataFrame],
    factor_ids: list[str],
) -> pd.DataFrame:
    """Long feature frame: index ``(timestamp, ticker)``, columns ``factor_ids``.

    Fast path: when every factor frame shares the same ``(index, columns)`` — the
    common case, since all signals are computed on the same panel — the long
    matrix is built by raveling each ``(time × ticker)`` array in C-order (which
    matches ``MultiIndex.from_product([index, columns])``), skipping pandas'
    slow per-frame ``.stack()``.  Falls back to the alignment-safe stack path
    when the frames differ in shape/labels.
    """
    if not factor_ids:
        return pd.DataFrame()
    ref = factor_signals[factor_ids[0]]
    ref_index, ref_cols = ref.index, ref.columns
    aligned = all(
        factor_signals[fid].index.equals(ref_index)
        and factor_signals[fid].columns.equals(ref_cols)
        for fid in factor_ids
    )
    if aligned:
        mi = pd.MultiIndex.from_product([ref_index, ref_cols])
        mat = np.empty((len(ref_index) * len(ref_cols), len(factor_ids)), dtype=float)
        for j, fid in enumerate(factor_ids):
            mat[:, j] = factor_signals[fid].to_numpy(dtype=float).ravel()
        X = pd.DataFrame(mat, index=mi, columns=list(factor_ids))
    else:
        cols = {fid: _stack(factor_signals[fid]) for fid in factor_ids}
        X = pd.DataFrame(cols)
    return X.replace([np.inf, -np.inf], np.nan)


def build_training_matrix(
    factor_signals: dict[str, pd.DataFrame],
    factor_ids: list[str],
    close: pd.DataFrame,
    target_horizon: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` for fitting.

    ``y`` is the ``target_horizon``-bar forward return for each ``(timestamp,
    ticker)`` sample.  A sample is kept when its target is finite **and at least
    one feature is present**; any remaining missing feature is imputed to ``0``
    — the neutral value for the cross-sectionally z-scored signals (a factor
    with no reading on a name simply abstains).

    This deliberately avoids ``dropna(how="any")`` across all factors: with many
    factors, each carrying its own NaN footprint (warm-up bars, or genuinely
    sparse alphas that only fire under specific conditions), the joint
    complete-case intersection collapses to a handful of rows — or zero — which
    silently destroys the model.

    ``max_samples`` (default: :data:`DEFAULT_MAX_TRAIN_SAMPLES`, overridable via
    ``MODELING_MAX_TRAIN_SAMPLES``) caps how many rows reach the estimator: when
    the kept set is larger it is uniformly subsampled with a fixed seed.  The
    same cap applied identically across preruns and models keeps a comparison
    fair while bounding fit time; prediction is never subsampled.
    """
    X = stack_features(factor_signals, factor_ids)
    y = _stack(forward_returns(close, horizon=target_horizon)).reindex(X.index)
    y = y.replace([np.inf, -np.inf], np.nan)

    feats = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)
    keep = np.isfinite(target) & ~np.isnan(feats).all(axis=1)
    X_kept = np.where(np.isnan(feats[keep]), 0.0, feats[keep])  # impute neutral
    y_kept = target[keep]

    cap = max_samples if max_samples is not None else _max_train_samples()
    if cap and len(y_kept) > cap:
        sel = np.random.default_rng(0).choice(len(y_kept), size=cap, replace=False)
        sel.sort()
        X_kept, y_kept = X_kept[sel], y_kept[sel]
    return X_kept, y_kept


def predict_to_signal(
    estimator,
    factor_signals: dict[str, pd.DataFrame],
    factor_ids: list[str],
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    """Run ``estimator.predict`` over the panel → ``(time × ticker)`` signal.

    Predicts on every ``(timestamp, ticker)`` with **at least one present
    feature**, imputing missing features to ``0`` — identically to
    :func:`build_training_matrix` — so the model takes a position wherever any
    factor has a reading.  Rows with no feature at all stay NaN (no position).
    The result is reindexed onto the requested ``index``/``columns`` so it lines
    up with the backtester's frames.
    """
    X = stack_features(factor_signals, factor_ids)
    feats = X.to_numpy(dtype=float)
    mask = ~np.isnan(feats).all(axis=1)
    preds = np.full(len(X), np.nan, dtype=float)
    if mask.any():
        Xi = np.where(np.isnan(feats[mask]), 0.0, feats[mask])  # impute neutral
        preds[mask] = estimator.predict(Xi)
    signal = pd.Series(preds, index=X.index).unstack()
    return signal.reindex(index=index, columns=columns)
