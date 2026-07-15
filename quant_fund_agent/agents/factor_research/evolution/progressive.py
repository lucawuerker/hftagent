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
) -> list[GenerationWindow]:
    """Compute the per-generation reveal schedule (see the module docstring).

    ``index`` is the panel ``DatetimeIndex`` (after any ``cutoff_date`` slicing).
    Raises ``ValueError`` for any illegal configuration (out-of-range fractions, a
    degenerate block length, or a generation-0 window that cannot keep ≥ 30 IS bars).
    """
    if generations < 1:
        raise ValueError(f"generations must be ≥ 1 (got {generations}).")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1) (got {test_frac}).")
    if not (0.0 < seed_frac < 1.0):
        raise ValueError(f"seed_frac must be in (0, 1) (got {seed_frac}).")
    if reveal_every < 1:
        raise ValueError(f"reveal_every must be ≥ 1 (got {reveal_every}).")
    if val_blocks < 1:
        raise ValueError(f"val_blocks must be ≥ 1 (got {val_blocks}).")

    n = len(index)
    test_start = int(math.floor(n * (1.0 - test_frac)))
    dev_len = test_start
    seed_end = int(math.floor(dev_len * seed_frac))

    reveal_gens = set(reveal_generations(generations, reveal_every))
    n_reveals = len(reveal_gens)
    remaining = dev_len - seed_end
    block_len = (remaining // n_reveals) if n_reveals else 0
    if block_len < 1:
        raise ValueError(
            f"block_len {block_len} < 1 — the dev window ({dev_len} bars) with "
            f"seed_end {seed_end} leaves too few bars to split into {n_reveals} "
            f"reveal blocks.  Lower seed_frac / reveal_every or lengthen the panel.")

    val_span = val_blocks * block_len

    def cumulative_end(k: int) -> int:
        """Dev frontier after ``k`` blocks revealed (the last block absorbs the
        remainder so ``k == n_reveals`` reaches the whole dev window)."""
        return dev_len if k >= n_reveals else seed_end + k * block_len

    def make(gen: int, k: int, reveal: bool) -> GenerationWindow:
        visible_end = cumulative_end(k)
        val_start = visible_end - val_span
        if val_start < MIN_IS_BARS:
            val_start = MIN_IS_BARS          # clamp: keep ≥ 30 IS bars (shrink VAL)
        block_bounds = None
        if reveal:
            prev_end = cumulative_end(k - 1)
            block_bounds = (index[prev_end].isoformat(), index[visible_end].isoformat())
        return GenerationWindow(
            generation=gen,
            visible_end=visible_end,
            val_start=val_start,
            is_end_ts=index[val_start].isoformat(),
            val_end_ts=index[visible_end].isoformat(),
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
    return windows
