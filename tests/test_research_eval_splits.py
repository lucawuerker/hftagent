"""Tests for the research-time temporal splitting (P0 foundation).

Covers the three-way IS/VAL/TEST split, CPCV purge+embargo, and walk-forward folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_fund_agent.research_eval.splits import (
    CPCVFold,
    ThreeWaySplit,
    cpcv_folds,
    n_cpcv_paths,
    three_way_split,
    walk_forward_folds,
    _purge_embargo_mask,
)


# ── three-way split ───────────────────────────────────────────────────────────

def test_three_way_fraction_split_partitions_timeline():
    split = three_way_split(100, is_frac=0.6, val_frac=0.2)
    assert isinstance(split, ThreeWaySplit)
    assert split.sizes == {"is": 60, "val": 20, "test": 20}
    # disjoint and exhaustive
    total = split.is_mask.astype(int) + split.val_mask.astype(int) + split.test_mask.astype(int)
    assert (total == 1).all()
    # ordered in time: IS before VAL before TEST
    assert np.flatnonzero(split.is_mask).max() < np.flatnonzero(split.val_mask).min()
    assert np.flatnonzero(split.val_mask).max() < np.flatnonzero(split.test_mask).min()
    assert (split.is_val_mask == (split.is_mask | split.val_mask)).all()


def test_three_way_calendar_split():
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    split = three_way_split(idx, is_end="2024-07-01", val_end="2024-10-01")
    assert idx[split.is_mask].max() < pd.Timestamp("2024-07-01")
    assert idx[split.val_mask].min() >= pd.Timestamp("2024-07-01")
    assert idx[split.val_mask].max() < pd.Timestamp("2024-10-01")
    assert idx[split.test_mask].min() >= pd.Timestamp("2024-10-01")


def test_three_way_split_validation():
    with pytest.raises(ValueError, match="is_frac"):
        three_way_split(100, is_frac=0.8, val_frac=0.3)  # sums to > 1
    with pytest.raises(ValueError, match="BOTH"):
        three_way_split(pd.date_range("2024-01-01", periods=10), is_end="2024-01-05")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        three_way_split(100, is_end="2024-07-01", val_end="2024-10-01")


# ── purge + embargo primitive ─────────────────────────────────────────────────

def test_purge_removes_horizon_overlap():
    n = 20
    test = np.zeros(n, dtype=bool)
    test[10:13] = True  # a test block at positions 10,11,12
    allowed = _purge_embargo_mask(test, horizon=2, embargo=0)
    # positions whose label [i, i+2] reaches into the test block must be dropped:
    # i=8 → [8,10] overlaps; i=9 → [9,11] overlaps. i=7 → [7,9] does not.
    assert not allowed[8] and not allowed[9]
    assert allowed[7]
    assert not allowed[10:13].any()  # test rows themselves never train


def test_embargo_removes_trailing_rows():
    n = 20
    test = np.zeros(n, dtype=bool)
    test[10:13] = True
    allowed = _purge_embargo_mask(test, horizon=0, embargo=2)
    # embargo drops the 2 rows immediately after the test block: 13, 14.
    assert not allowed[13] and not allowed[14]
    assert allowed[15]


# ── CPCV ──────────────────────────────────────────────────────────────────────

def test_cpcv_fold_count_and_disjointness():
    folds = cpcv_folds(60, n_groups=6, k_test=2, horizon=1, embargo=0)
    from math import comb
    assert len(folds) == comb(6, 2) == 15
    for f in folds:
        assert isinstance(f, CPCVFold)
        assert not (f.train & f.test).any()          # never train on test rows
        assert f.test.sum() > 0 and f.train.sum() > 0


def test_cpcv_purge_shrinks_train():
    no_purge = cpcv_folds(60, n_groups=6, k_test=2, horizon=0, embargo=0)
    purged = cpcv_folds(60, n_groups=6, k_test=2, horizon=5, embargo=3)
    # with a horizon + embargo the training set can only be smaller
    for a, b in zip(no_purge, purged):
        assert b.train.sum() <= a.train.sum()
    assert sum(b.train.sum() for b in purged) < sum(a.train.sum() for a in no_purge)


def test_cpcv_respects_mask():
    split = three_way_split(120, is_frac=0.6, val_frac=0.2)  # TEST is the tail 20%
    folds = cpcv_folds(120, n_groups=5, k_test=2, mask=split.is_val_mask)
    for f in folds:
        # nothing in TEST is ever used for train or test in a CPCV fold
        assert not (f.train & split.test_mask).any()
        assert not (f.test & split.test_mask).any()


def test_cpcv_path_count():
    assert n_cpcv_paths(6, 2) == 5      # C(6,2)·2/6 = 15·2/6
    assert n_cpcv_paths(10, 2) == 9     # C(10,2)·2/10 = 45·2/10


def test_cpcv_validation():
    with pytest.raises(ValueError, match="k_test"):
        cpcv_folds(60, n_groups=5, k_test=5)
    with pytest.raises(ValueError, match="n_groups"):
        cpcv_folds(60, n_groups=1, k_test=1)


# ── walk-forward ──────────────────────────────────────────────────────────────

def test_walk_forward_anchored_is_expanding():
    folds = walk_forward_folds(120, n_folds=4, horizon=1, embargo=0, anchored=True)
    assert len(folds) == 4
    sizes = [f.train.sum() for f in folds]
    assert sizes == sorted(sizes)                     # expanding training window
    for f in folds:
        assert not (f.train & f.test).any()
        # test always strictly after the training window (walk forward)
        assert np.flatnonzero(f.train).max() < np.flatnonzero(f.test).min()


def test_walk_forward_rolling_is_constant_ish():
    folds = walk_forward_folds(120, n_folds=4, anchored=False)
    sizes = [f.train.sum() for f in folds]
    # rolling windows are single blocks → roughly constant, and much smaller than the
    # anchored final fold
    assert max(sizes) - min(sizes) <= 2
