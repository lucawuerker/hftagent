"""Progressive data reveal — the dev window is revealed to the search as a stream.

Over generations, selection pressure adaptively fits the population to whatever VAL
window it is scored on ("the ratchet") — a process-level overfitting no per-candidate
statistic on that *same reused* window can police (the statistic becomes the
optimisation target).  Progressive reveal *prevents* the ratchet rather than
detecting it: the development window is revealed block-by-block across generations, so
part of every generation's scoring window has never been queried by any earlier
selection.

Semantics (fixed; see ``docs/research-evolution/PROGRESSIVE_REVEAL_PLAN.md``):

* Let the (cutoff-sliced) panel index have ``n`` bars.  ``test_start =
  floor(n·(1−test_frac))`` and **dev** = ``[0, test_start)``.  The tail
  ``[test_start, n)`` is the final TEST — never revealed, never scored in-run
  (exactly today's TEST semantics).
* ``seed_end = floor(dev_len·seed_frac)`` is the window visible at generation 0.
* A generation ``g ∈ {1..G}`` is a **reveal generation** iff
  ``(g−1) % reveal_every == 0`` (so generation 1 always reveals).  With ``R`` reveal
  generations, the remaining ``dev_len − seed_end`` bars are split into ``R`` equal
  blocks (remainder bars go to the **last** block), so the final reveal makes the
  whole dev window visible.
* The window is **expanding** — blocks are never dropped (old blocks stay as
  permanent constraints; a rolling cap was explicitly rejected).
* ``holdout_last=True`` (the ``--final-holdout`` flag) splits the remaining dev
  bars into ``R+1`` blocks instead of ``R`` and keeps the last one hidden from
  every generation: the run's reveal cadence is unchanged but its frontier never
  reaches ``dev_len``.  A **terminal** window (``generation = G+1``, reveal=True,
  ``visible_end = dev_len``) is appended for the loop's post-run rescore-only
  step, so the final front is ranked on a block no selection pressure ever
  queried (otherwise the last revealed block is both selected-on and ranked-on,
  and the front tilts toward it).
* **Sliding VAL** = the last ``val_blocks`` blocks: ``val_span = val_blocks·block_len``,
  ``val_start = visible_end − val_span``, IS = ``[0, val_start)``, VAL =
  ``[val_start, visible_end)``.  IS is clamped to keep ≥ 30 bars (the harness's
  minimum fit support); if the clamp binds, ``val_span`` shrinks.

The output is a list of :class:`GenerationWindow`, one per generation ``0..G`` (so
``schedule[g]`` is the window in force at generation ``g``).  It is a **pure function**
of ``(index, config)`` — no wall clock, no RNG — so a resumed run recomputes the exact
same schedule from ``controller.generation``.

The timestamp convention matches ``three_way_split``'s calendar mode (IS = ``idx <
is_end``, VAL = ``[is_end, val_end)``): ``is_end_ts = index[val_start]`` and
``val_end_ts = index[visible_end]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The harness's minimum in-sample fit support (see ``_combined_prediction``): IS must
# never shrink below this many bars, or the marginal-value model cannot be fit.
MIN_IS_BARS = 30


@dataclass(frozen=True)
class GenerationWindow:
    """The IS/VAL frontier in force at one generation of a progressive-reveal run."""

    generation: int
    visible_end: int                        # dev frontier: bar index (exclusive)
    val_start: int                          # VAL start: bar index
    is_end_ts: str                          # ISO timestamp = index[val_start]
    val_end_ts: str                         # ISO timestamp = index[visible_end]
    reveal: bool                            # True if this generation revealed a block
    block_bounds: tuple[str, str] | None    # [start_ts, end_ts) of the new block


def reveal_generations(generations: int, reveal_every: int) -> list[int]:
    """The generations ``g ∈ {1..generations}`` that reveal a new block."""
    return [g for g in range(1, generations + 1) if (g - 1) % reveal_every == 0]


def build_schedule(
    index,
    *,
    generations: int,
    test_frac: float,
    seed_frac: float,
    reveal_every: int,
    val_blocks: int,
    holdout_last: bool = False,
    wf_blocks: int = 0,
    wf_block_bars: int = 126,
) -> list[GenerationWindow]:
    """Compute the per-generation reveal schedule (see the module docstring).

    ``index`` is the panel ``DatetimeIndex`` (after any ``cutoff_date`` slicing).
    Raises ``ValueError`` for any illegal configuration (out-of-range fractions, a
    degenerate block length, or a generation-0 window that cannot keep ≥ 30 IS bars).

    ``wf_blocks > 0`` switches to the TWO-PHASE walk-forward schedule (final-run
    decision 2026-08-01): the LAST ``wf_blocks`` generations each reveal exactly one
    fixed-size block of ``wf_block_bars`` bars covering the tail of the dev window
    (the "live walk-forward" phase — each block is prequentially scored before the
    archive may adapt to it), while generations ``1..G−wf_blocks`` reveal the earlier
    span ``[seed_end, wf_start)`` in equal blocks at the ``reveal_every`` cadence
    (``seed_frac`` is the fraction of the PHASE-1 span visible at generation 0).
    Incompatible with ``holdout_last``.
    """
    if generations < 1:
        raise ValueError(f"generations must be ≥ 1 (got {generations}).")
    if not (0.0 <= test_frac < 1.0):
        # test_frac == 0 is legal: NO in-panel TEST tail — the panel's end IS
        # the dev frontier and the only held-out data is whatever lies beyond
        # the config's end date (the 2-year forward reserve, per the final-run
        # decision 2026-07-29: a single forward-validation frame).
        raise ValueError(f"test_frac must be in [0, 1) (got {test_frac}).")
    if not (0.0 < seed_frac < 1.0):
        raise ValueError(f"seed_frac must be in (0, 1) (got {seed_frac}).")
    if reveal_every < 1:
        raise ValueError(f"reveal_every must be ≥ 1 (got {reveal_every}).")
    if val_blocks < 1:
        raise ValueError(f"val_blocks must be ≥ 1 (got {val_blocks}).")
    if wf_blocks:
        return _build_two_phase_schedule(
            index, generations=generations, test_frac=test_frac,
            seed_frac=seed_frac, reveal_every=reveal_every, val_blocks=val_blocks,
            holdout_last=holdout_last, wf_blocks=wf_blocks,
            wf_block_bars=wf_block_bars)

    n = len(index)
    test_start = int(math.floor(n * (1.0 - test_frac)))
    dev_len = test_start
    seed_end = int(math.floor(dev_len * seed_frac))

    reveal_gens = set(reveal_generations(generations, reveal_every))
    n_reveals = len(reveal_gens)
    # --final-holdout: one extra block that no generation ever sees; it is only
    # revealed by the terminal (G+1) rescore-only window appended below.
    n_blocks = n_reveals + (1 if holdout_last else 0)
    remaining = dev_len - seed_end
    block_len = (remaining // n_blocks) if n_blocks else 0
    if block_len < 1:
        raise ValueError(
            f"block_len {block_len} < 1 — the dev window ({dev_len} bars) with "
            f"seed_end {seed_end} leaves too few bars to split into {n_blocks} "
            f"reveal blocks.  Lower seed_frac / reveal_every or lengthen the panel.")

    val_span = val_blocks * block_len

    def cumulative_end(k: int) -> int:
        """Dev frontier after ``k`` blocks revealed (the last block absorbs the
        remainder so ``k == n_blocks`` reaches the whole dev window)."""
        return dev_len if k >= n_blocks else seed_end + k * block_len

    def _ts(pos: int) -> str:
        """EXCLUSIVE-bound timestamp for position ``pos``.

        For ``pos == len(index)`` (test_frac == 0: the dev frontier reaches the
        very end of the panel) there is no bar to point at — return one second
        past the final bar so ``< frontier`` comparisons include it.
        """
        if pos >= len(index):
            import pandas as pd  # local: this module is otherwise pandas-free
            return (index[-1] + pd.Timedelta(seconds=1)).isoformat()
        return index[pos].isoformat()

    def make(gen: int, k: int, reveal: bool) -> GenerationWindow:
        visible_end = cumulative_end(k)
        val_start = visible_end - val_span
        if val_start < MIN_IS_BARS:
            val_start = MIN_IS_BARS          # clamp: keep ≥ 30 IS bars (shrink VAL)
        block_bounds = None
        if reveal:
            prev_end = cumulative_end(k - 1)
            block_bounds = (_ts(prev_end), _ts(visible_end))
        return GenerationWindow(
            generation=gen,
            visible_end=visible_end,
            val_start=val_start,
            is_end_ts=_ts(val_start),
            val_end_ts=_ts(visible_end),
            reveal=reveal,
            block_bounds=block_bounds,
        )

    gen0 = make(0, 0, reveal=False)
    if gen0.val_start < MIN_IS_BARS or (gen0.visible_end - gen0.val_start) < 1:
        raise ValueError(
            f"generation-0 window is degenerate (IS={gen0.val_start}, "
            f"VAL={gen0.visible_end - gen0.val_start}); need IS ≥ {MIN_IS_BARS} and "
            f"VAL ≥ 1.  Raise seed_frac or lengthen the panel.")

    windows = [gen0]
    k = 0
    for g in range(1, generations + 1):
        reveal = g in reveal_gens
        if reveal:
            k += 1
        windows.append(make(g, k, reveal=reveal))
    if holdout_last:
        # Terminal rescore-only window: reveals the held-out block AFTER the
        # final generation (``schedule[generations + 1]``).
        windows.append(make(generations + 1, n_blocks, reveal=True))
    return windows


def _build_two_phase_schedule(
    index,
    *,
    generations: int,
    test_frac: float,
    seed_frac: float,
    reveal_every: int,
    val_blocks: int,
    holdout_last: bool,
    wf_blocks: int,
    wf_block_bars: int,
) -> list[GenerationWindow]:
    """The two-phase walk-forward schedule (``build_schedule`` with ``wf_blocks > 0``).

    Layout over the dev window ``[0, dev_len)``::

        [0 .. seed_end)                      visible at generation 0
        [seed_end .. wf_start)               phase 1: equal blocks, one per reveal
                                             generation in ``1..G−wf_blocks``
                                             (``reveal_every`` cadence)
        [wf_start .. dev_len)                phase 2: the last ``wf_blocks``
                                             generations reveal one fixed
                                             ``wf_block_bars``-bar block each

    ``seed_end = floor(wf_start · seed_frac)`` — the seed fraction is of the
    PHASE-1 span, so the walk-forward tail never shrinks the fitting window.
    The sliding VAL keeps its meaning (last ``val_blocks`` revealed blocks; block
    lengths are heterogeneous across the phase boundary).
    """
    if holdout_last:
        raise ValueError("wf_blocks is incompatible with --final-holdout: the "
                         "walk-forward tail already ends at the panel edge and "
                         "every block is prequentially scored before adaptation.")
    if wf_blocks < 1:
        raise ValueError(f"wf_blocks must be ≥ 1 (got {wf_blocks}).")
    if wf_block_bars < 1:
        raise ValueError(f"wf_block_bars must be ≥ 1 (got {wf_block_bars}).")
    if generations < wf_blocks:
        raise ValueError(
            f"generations ({generations}) must be ≥ wf_blocks ({wf_blocks}) — the "
            f"last wf_blocks generations each reveal one walk-forward block.")

    n = len(index)
    dev_len = int(math.floor(n * (1.0 - test_frac)))
    wf_start = dev_len - wf_blocks * wf_block_bars
    if wf_start < MIN_IS_BARS + 1:
        raise ValueError(
            f"walk-forward span {wf_blocks}·{wf_block_bars} bars leaves only "
            f"{wf_start} phase-1 bars (dev {dev_len}); need > {MIN_IS_BARS}.  "
            f"Shrink wf_blocks/wf_block_bars or lengthen the panel.")

    p1_gens = generations - wf_blocks
    p1_reveal_gens = [g for g in range(1, p1_gens + 1) if (g - 1) % reveal_every == 0]
    r1 = len(p1_reveal_gens)
    if r1:
        seed_end = int(math.floor(wf_start * seed_frac))
        block1 = (wf_start - seed_end) // r1
        if block1 < 1:
            raise ValueError(
                f"phase-1 block_len {block1} < 1 — phase-1 span {wf_start} bars with "
                f"seed_end {seed_end} cannot be split into {r1} reveal blocks.")
    else:
        # No phase-1 generations: the whole pre-walk-forward span is the seed.
        seed_end = wf_start
        block1 = wf_block_bars

    # Cumulative frontier after k blocks revealed: r1 phase-1 blocks (last absorbs
    # the remainder so block r1 lands exactly on wf_start), then wf_blocks fixed ones.
    ends = [seed_end]
    for k in range(1, r1 + 1):
        ends.append(wf_start if k == r1 else seed_end + k * block1)
    for j in range(1, wf_blocks + 1):
        ends.append(wf_start + j * wf_block_bars)   # ends[-1] == dev_len

    def _ts(pos: int) -> str:
        if pos >= len(index):
            import pandas as pd  # local: this module is otherwise pandas-free
            return (index[-1] + pd.Timedelta(seconds=1)).isoformat()
        return index[pos].isoformat()

    def make(gen: int, k: int, reveal: bool) -> GenerationWindow:
        visible_end = ends[k]
        if k >= val_blocks:
            val_start = ends[k - val_blocks]
        else:
            # Before val_blocks blocks exist, extend virtually into the seed window
            # at the phase-1 block length (mirrors the classic constant val_span):
            # VAL = the k real blocks plus (val_blocks − k) virtual seed blocks.
            val_start = ends[0] - (val_blocks - k) * block1
        if val_start < MIN_IS_BARS:
            val_start = MIN_IS_BARS
        block_bounds = None
        if reveal:
            block_bounds = (_ts(ends[k - 1]), _ts(visible_end))
        return GenerationWindow(
            generation=gen,
            visible_end=visible_end,
            val_start=val_start,
            is_end_ts=_ts(val_start),
            val_end_ts=_ts(visible_end),
            reveal=reveal,
            block_bounds=block_bounds,
        )

    gen0 = make(0, 0, reveal=False)
    if gen0.val_start < MIN_IS_BARS or (gen0.visible_end - gen0.val_start) < 1:
        raise ValueError(
            f"generation-0 window is degenerate (IS={gen0.val_start}, "
            f"VAL={gen0.visible_end - gen0.val_start}); need IS ≥ {MIN_IS_BARS} and "
            f"VAL ≥ 1.  Raise seed_frac or lengthen the panel.")

    windows = [gen0]
    k = 0
    p1_reveals = set(p1_reveal_gens)
    for g in range(1, generations + 1):
        reveal = g in p1_reveals or g > p1_gens
        if reveal:
            k += 1
        windows.append(make(g, k, reveal=reveal))
    return windows
