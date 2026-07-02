"""Temporal splitting for research-time overfitting control.

The evolutionary search is a multiple-testing machine: generate thousands of
candidates, keep the best OOS score, and you have merely moved data-snooping up a
level.  The overfit protocol in ``docs/research-evolution/DESIGN.md`` therefore
threads *one* splitting seam through the whole harness, exactly like ``cutoff_date``
does in the deployed fund, so research and deployment cannot drift:

* **Three-way temporal split** (:func:`three_way_split`).  ``IS`` fits the ML
  ensemble used for marginal-value scoring; ``VALIDATION`` computes the fitness and
  is *deliberately burned* by the search; ``TEST`` is touched **once**, by the
  Statistician, on the final Pareto survivors only.
* **CPCV** (:func:`cpcv_folds`) — Combinatorial Purged Cross-Validation over
  ``IS∪VAL``: combinatorial train/test groups with **purging** (drop training
  labels whose forward-return horizon overlaps a test block) and **embargo** (a
  buffer of bars after each test block).  The output is a *distribution* of OOS
  scores per candidate, not a single point — the development-time fitness engine.
* **Walk-forward** (:func:`walk_forward_folds`) — sequential expanding/rolling
  folds; the whole loop is re-run period-by-period inside this wrapper for the
  thesis results chapter, so every discovery is OOS to the next period.

Everything operates on *integer positions* over a timeline of length ``n`` (or a
pandas ``Index``), returning boolean masks, so it is agnostic to whether the
timeline is daily SP100 bars or intraday LOBSTER bars.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _length(index_or_n: Any) -> int:
    """Number of timeline positions from either a length or an index-like object."""
    if isinstance(index_or_n, (int, np.integer)):
        return int(index_or_n)
    return len(index_or_n)


def _purge_embargo_mask(test_mask: np.ndarray, horizon: int, embargo: int) -> np.ndarray:
    """Allowed-train mask given a ``test_mask``, applying purge + embargo.

    A training observation ``i`` carries a forward-return label spanning
    ``[i, i+horizon]``.  It is **purged** if any test position falls in that window
    (its label overlaps the test block), and **embargoed** if a test position falls
    in ``[i-embargo, i-1]`` (it sits within ``embargo`` bars *after* a test block,
    where serial correlation could leak).  The returned mask is the set of training
    positions that survive both filters (test positions are always excluded).
    """
    n = len(test_mask)
    horizon = max(0, int(horizon))
    embargo = max(0, int(embargo))

    # future[i] = a test position lies in [i, i+horizon] → i's label overlaps a test block.
    future = np.zeros(n, dtype=bool)
    for lag in range(0, horizon + 1):
        if lag == 0:
            future |= test_mask
        else:
            future[: n - lag] |= test_mask[lag:]

    # past[i] = a test position lies in [i-embargo, i-1] → i is within the embargo tail.
    past = np.zeros(n, dtype=bool)
    for lag in range(1, embargo + 1):
        past[lag:] |= test_mask[: n - lag]

    return (~test_mask) & (~future) & (~past)


# ── three-way IS / VALIDATION / TEST split ───────────────────────────────────

@dataclass(frozen=True)
class ThreeWaySplit:
    """Boolean masks for the IS / VALIDATION / TEST temporal split.

    ``is_val_mask`` is the convenience union ``IS ∪ VAL`` over which CPCV runs
    (the development-time data the search is allowed to burn).
    """

    is_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def is_val_mask(self) -> np.ndarray:
        return self.is_mask | self.val_mask

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "is": int(self.is_mask.sum()),
            "val": int(self.val_mask.sum()),
            "test": int(self.test_mask.sum()),
        }


def three_way_split(
    index_or_n: Any,
    *,
    is_frac: float = 0.6,
    val_frac: float = 0.2,
    is_end: str | datetime | None = None,
    val_end: str | datetime | None = None,
) -> ThreeWaySplit:
    """Contiguous IS → VAL → TEST split of a timeline.

    Two modes, exactly like ``ComparisonConfig.split_masks``:

    * **fraction** (default) — the first ``is_frac`` of the bars are IS, the next
      ``val_frac`` are VALIDATION, the remainder is TEST.  ``is_frac + val_frac``
      must be < 1 so TEST is non-empty.
    * **calendar** — pass ``is_end`` and ``val_end`` (ISO timestamps or datetimes);
      requires ``index_or_n`` be a ``DatetimeIndex``.  IS is every bar strictly
      before ``is_end``, VAL is ``[is_end, val_end)``, TEST is ``≥ val_end``.
    """
    if (is_end is None) != (val_end is None):
        raise ValueError("Calendar three-way split needs BOTH is_end and val_end.")

    if is_end is not None:
        if isinstance(index_or_n, (int, np.integer)):
            raise ValueError("Calendar split needs a DatetimeIndex, not a length.")
        idx = index_or_n
        lo = datetime.fromisoformat(is_end) if isinstance(is_end, str) else is_end
        hi = datetime.fromisoformat(val_end) if isinstance(val_end, str) else val_end
        if not lo < hi:
            raise ValueError(f"is_end ({lo}) must precede val_end ({hi}).")
        is_mask = np.asarray(idx < lo)
        val_mask = np.asarray((idx >= lo) & (idx < hi))
        test_mask = np.asarray(idx >= hi)
        return ThreeWaySplit(is_mask, val_mask, test_mask)

    if is_frac <= 0 or val_frac <= 0 or is_frac + val_frac >= 1.0:
        raise ValueError(
            f"Need 0 < is_frac, val_frac and is_frac+val_frac < 1 "
            f"(got is_frac={is_frac}, val_frac={val_frac}).")
    n = _length(index_or_n)
    is_cut = int(n * is_frac)
    val_cut = int(n * (is_frac + val_frac))
    is_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    is_mask[:is_cut] = True
    val_mask[is_cut:val_cut] = True
    test_mask[val_cut:] = True
    return ThreeWaySplit(is_mask, val_mask, test_mask)


# ── Combinatorial Purged Cross-Validation ────────────────────────────────────

@dataclass(frozen=True)
class CPCVFold:
    """One CPCV split: a purged-embargoed train mask and its test mask.

    ``test_groups`` records which of the ``n_groups`` contiguous time groups make up
    this fold's test set (for logging / path reconstruction).
    """

    train: np.ndarray
    test: np.ndarray
    test_groups: tuple[int, ...]


def _contiguous_groups(n: int, n_groups: int) -> list[np.ndarray]:
    """Split ``range(n)`` into ``n_groups`` contiguous (near-equal) position blocks."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be ≥ 2 (got {n_groups}).")
    if n_groups > n:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed the timeline length ({n}).")
    return [g for g in np.array_split(np.arange(n), n_groups) if len(g)]


def cpcv_folds(
    index_or_n: Any,
    *,
    n_groups: int = 6,
    k_test: int = 2,
    horizon: int = 1,
    embargo: int = 0,
    mask: np.ndarray | None = None,
) -> list[CPCVFold]:
    """Combinatorial Purged Cross-Validation folds over a timeline.

    Partition the timeline into ``n_groups`` contiguous groups, then for every
    combination of ``k_test`` groups form a fold whose test set is their union and
    whose train set is the remaining positions with :func:`_purge_embargo_mask`
    applied at the given forward-return ``horizon`` and ``embargo``.  There are
    ``C(n_groups, k_test)`` folds — a *distribution* of OOS scores per candidate.

    ``mask`` (optional) restricts CPCV to a sub-timeline (e.g. ``IS∪VAL`` from a
    :class:`ThreeWaySplit`): grouping happens over the masked positions only, and
    the returned train/test masks are full-length with everything outside ``mask``
    set to ``False``.
    """
    n = _length(index_or_n)
    if k_test < 1 or k_test >= n_groups:
        raise ValueError(f"Need 1 ≤ k_test < n_groups (got k_test={k_test}, n_groups={n_groups}).")

    active = np.flatnonzero(mask) if mask is not None else np.arange(n)
    groups = _contiguous_groups(len(active), n_groups)  # index into `active`

    folds: list[CPCVFold] = []
    for combo in itertools.combinations(range(len(groups)), k_test):
        test_mask = np.zeros(n, dtype=bool)
        for gi in combo:
            test_mask[active[groups[gi]]] = True
        train_mask = _purge_embargo_mask(test_mask, horizon, embargo)
        if mask is not None:
            train_mask &= mask  # never train on TEST-set (out-of-scope) rows
        folds.append(CPCVFold(train=train_mask, test=test_mask, test_groups=tuple(combo)))
    return folds


def n_cpcv_paths(n_groups: int, k_test: int) -> int:
    """Number of distinct backtest *paths* CPCV reconstructs = ``C(N,k)·k / N``.

    (Each of the ``C(N,k)`` folds tests ``k`` groups; summed over folds every group
    is tested the same number of times, which stitches into this many end-to-end
    OOS paths — the López de Prado path count.)
    """
    return math.comb(n_groups, k_test) * k_test // n_groups


# ── walk-forward ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WalkForwardFold:
    """One walk-forward fold: an (expanding or rolling) train mask + its test mask."""

    train: np.ndarray
    test: np.ndarray
    fold: int


def walk_forward_folds(
    index_or_n: Any,
    *,
    n_folds: int = 5,
    horizon: int = 1,
    embargo: int = 0,
    anchored: bool = True,
    mask: np.ndarray | None = None,
) -> list[WalkForwardFold]:
    """Sequential walk-forward folds stepping forward in time.

    The timeline is cut into ``n_folds + 1`` contiguous blocks; fold ``f`` tests on
    block ``f+1`` and trains on blocks ``0..f`` when ``anchored`` (expanding window,
    the default) or on block ``f`` alone when not (rolling window).  Purge + embargo
    are applied between each train and its test block so a label never bleeds across
    the seam.  ``mask`` restricts to a sub-timeline as in :func:`cpcv_folds`.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be ≥ 1 (got {n_folds}).")
    n = _length(index_or_n)
    active = np.flatnonzero(mask) if mask is not None else np.arange(n)
    blocks = _contiguous_groups(len(active), n_folds + 1)  # index into `active`

    folds: list[WalkForwardFold] = []
    for f in range(n_folds):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[active[blocks[f + 1]]] = True
        if anchored:
            train_positions = np.concatenate([blocks[j] for j in range(f + 1)])
        else:
            train_positions = blocks[f]
        base_train = np.zeros(n, dtype=bool)
        base_train[active[train_positions]] = True
        allowed = _purge_embargo_mask(test_mask, horizon, embargo)
        train_mask = base_train & allowed
        folds.append(WalkForwardFold(train=train_mask, test=test_mask, fold=f))
    return folds
