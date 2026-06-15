"""Equivalence tests for the vectorised time-series operators in ``ops.py``.

The operators ``ts_argmax``, ``ts_argmin``, ``ts_rank``, ``product`` and
``decay_linear`` were re-implemented with NumPy sliding windows (replacing the
catastrophically slow ``rolling(n).apply(<python callable>)`` form — a single
``ts_rank(df, 60)`` on the LOBSTER panel went from ~80s to well under 1s).

These tests pin the new implementations to the *original* ``rolling.apply``
references on data containing NaNs and ties, across several window sizes, so the
speed-up provably changes no numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.factors import ops

WINDOWS = [5, 20, 60]


# ── original implementations, verbatim, as the ground truth ──────────────────

def _ref_ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(lambda x: float(np.argmax(x) + 1), raw=True)


def _ref_ts_argmin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(lambda x: float(np.argmin(x) + 1), raw=True)


def _ref_ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(
        lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False,
    )


def _ref_product(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).apply(np.prod, raw=True)


def _ref_decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return df.rolling(n, min_periods=n).apply(lambda x: np.dot(x, weights), raw=True)


_CASES = [
    (ops.ts_argmax, _ref_ts_argmax),
    (ops.ts_argmin, _ref_ts_argmin),
    (ops.ts_rank, _ref_ts_rank),
    (ops.product, _ref_product),
    (ops.decay_linear, _ref_decay_linear),
]


def _assert_equivalent(new: pd.DataFrame, ref: pd.DataFrame) -> None:
    assert new.shape == ref.shape
    assert new.index.equals(ref.index)
    assert new.columns.equals(ref.columns)
    a, b = new.to_numpy(dtype=float), ref.to_numpy(dtype=float)
    # identical NaN pattern (min_periods=n + NaN-in-window semantics)
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN masks differ"
    both = ~np.isnan(a)
    if both.any():
        assert np.nanmax(np.abs(a[both] - b[both])) < 1e-9


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    T, N = 300, 6
    arr = rng.standard_normal((T, N))
    # ties: round one column so ts_rank exercises the average-rank path
    arr[:, 0] = np.round(arr[:, 0], 1)
    # leading NaNs (as produced by upstream rolling ops) + scattered NaNs
    for j in range(N):
        arr[: 3 * j, j] = np.nan
    arr[rng.random((T, N)) < 0.05] = np.nan
    idx = pd.date_range("2020-01-01", periods=T, freq="10s")
    return pd.DataFrame(arr, index=idx, columns=[f"t{j}" for j in range(N)])


@pytest.mark.parametrize("new_fn,ref_fn", _CASES, ids=lambda f: getattr(f, "__name__", str(f)))
@pytest.mark.parametrize("n", WINDOWS)
def test_operator_matches_reference_synthetic(new_fn, ref_fn, n):
    df = _synthetic_frame()
    _assert_equivalent(new_fn(df, n), ref_fn(df, n))


@pytest.mark.parametrize("new_fn,ref_fn", _CASES, ids=lambda f: getattr(f, "__name__", str(f)))
def test_operator_matches_reference_no_nan(new_fn, ref_fn):
    """A clean (NaN-free) frame, full-overlap windows, all distinct values."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.standard_normal((120, 4)), columns=list("abcd"))
    _assert_equivalent(new_fn(df, 20), ref_fn(df, 20))


def test_window_larger_than_frame_all_nan():
    df = pd.DataFrame(np.arange(12.0).reshape(4, 3), columns=list("xyz"))
    out = ops.ts_rank(df, 10)  # n > T
    assert out.shape == df.shape
    assert out.isna().all().all()
