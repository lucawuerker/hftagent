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
already-normalised signals — the two paths build identical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.data_loader import forward_returns


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
    """Long feature frame: index ``(timestamp, ticker)``, columns ``factor_ids``."""
    cols = {fid: _stack(factor_signals[fid]) for fid in factor_ids}
    X = pd.DataFrame(cols)
    return X.replace([np.inf, -np.inf], np.nan)


def build_training_matrix(
    factor_signals: dict[str, pd.DataFrame],
    factor_ids: list[str],
    close: pd.DataFrame,
    target_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` for fitting, dropping any row with a NaN feature/target.

    ``y`` is the ``target_horizon``-bar forward return for each ``(timestamp,
    ticker)`` sample.
    """
    X = stack_features(factor_signals, factor_ids)
    y = _stack(forward_returns(close, horizon=target_horizon)).reindex(X.index)
    df = X.copy()
    df["__target__"] = y.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how="any")
    return df[factor_ids].to_numpy(dtype=float), df["__target__"].to_numpy(dtype=float)


def predict_to_signal(
    estimator,
    factor_signals: dict[str, pd.DataFrame],
    factor_ids: list[str],
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    """Run ``estimator.predict`` over the panel → ``(time × ticker)`` signal.

    Rows with any NaN feature are left as NaN (no position), exactly mirroring
    how the training matrix dropped them.  The result is reindexed onto the
    requested ``index``/``columns`` so it lines up with the backtester's frames.
    """
    X = stack_features(factor_signals, factor_ids)
    mask = X.notna().all(axis=1)
    preds = pd.Series(np.nan, index=X.index, dtype=float)
    if mask.any():
        preds.loc[mask] = estimator.predict(X.loc[mask].to_numpy(dtype=float))
    signal = preds.unstack()
    return signal.reindex(index=index, columns=columns)
