# Implementation plan: Pareto-vector rework + dev-wide residual IC + progressive data reveal

**Status: DECIDED, not yet implemented.** This document is a complete handoff:
give it to the implementing model verbatim as its task description.
**Amended 2026-07-16:** Change 4 added (temporal-degradation hard gate becomes
the fifth Pareto axis) — the final axis count is **five**, the gate count three.

---

## Prompt to the implementing model

Implement the four changes specified below in this repository, in the order
given in "Suggested order", exactly as specified. Do not implement anything in the "Out of scope"
section. Do not redesign: every design decision has already been made and is
recorded here with its rationale; where this document says "decide X like Y",
follow it literally.

Work rules:

- Python: always `./venv/bin/python` and `./venv/bin/pytest` (never system python).
- Tests run in-process by default (`QF_USE_MCP=0` path); the MCP client/service
  in-process seam must stay byte-identical to the subprocess path.
- Everything in the evolution stack must stay **deterministic**: no wall clock,
  no unseeded RNG, identical inputs → identical outputs (checkpoints must
  resume reproducibly).
- New run-level behavior defaults **OFF** so existing baseline/ablation arms are
  unchanged (Change 3 is flag-gated; Changes 1–2 are permanent design
  corrections and are NOT flag-gated).
- After each stage below, run the full test suite and fix every breakage before
  moving on. Line numbers given are anchors as of writing — grep the symbol if
  they have drifted.
- When done, update the docs listed in the "Docs" section (this is a project
  convention, not optional).

---

## Background (why these changes)

The evolutionary Factor Researcher scores candidates on a fixed VAL window every
generation. Over generations, selection pressure adaptively fits the population
to that fixed window ("the ratchet") — a process-level overfitting that no
per-candidate statistic computed on the same reused window can police (the
statistic itself becomes the optimization target). Three consequences, decided
with the author:

1. **The `robustness` (Probabilistic Sharpe Ratio) Pareto axis is removed.**
   It was meant to control this overfitting but cannot (see above). It is also
   technically flawed: (a) its marginal per-bar P&L uses *overlapping* h-bar
   forward returns, so consecutive bars share h−1 return bars and the PSR's
   `n_obs` overstates the effective sample ~h× (the comparison harness's
   tranche-book convention exists precisely to avoid this and is not used
   here); (b) its folded-in plateau/perturbation/sign adjustments are IC-scale
   numbers subtracted from a [0,1] probability — dimensionally incoherent;
   (c) at per-bar scale PSR ≈ Φ(SR·√n), a near-monotone transform of the
   primary marginal axis on the same window → a redundant axis that dilutes
   NSGA-II dominance.
2. **Residual IC scores on IS∪VAL.** Factor formulas have no fitted
   parameters; only the combiner and the orthogonalization betas are fitted.
   Scoring the residual IC on IS∪VAL (betas still fit on IS) quadruples the
   effective sample of the independence axis, lowering its noise floor.
3. **Progressive data reveal ("the run experiences data as a stream").** The
   dev window is revealed block-by-block across generations, so part of each
   generation's scoring window has never been queried by any earlier
   selection — prevention rather than detection of the ratchet. The window is
   **expanding** (never drops old blocks): more scored data = a higher
   noise-floor barrier against adaptive overfitting; old blocks remain
   permanent constraints (dropping them would re-introduce era-specialization);
   and the prequential OOS curve stays interpretable (score changes have one
   cause: the newly revealed block). A final `test_frac` tail (default 20%,
   author wants 15–20%) is **never revealed** and is only scored once after the
   whole run — unchanged from today's TEST semantics.
4. **The temporal-degradation hard gate becomes the fifth Pareto axis**
   (decided 2026-07-16). The gate (sign-aligned VAL/IS IC ratio ≥ τ) was too
   harsh: a conditioning factor whose standalone IC fluctuates around zero can
   show |IS IC| just above `min_is_ic` with the opposite sign on VAL and be
   excluded outright — even though its value is *marginal*, the very factor
   class the primary axis exists to protect. As a Pareto axis, temporal
   inconsistency must be traded off against the other objectives instead of
   being fatal. Under progressive reveal the axis also gains meaning: at
   reveal generations part of VAL was never scored by any earlier selection,
   so the ratio measured there reflects retention on genuinely unseen data —
   which is why the reveal-generation diagnostic stamp (3d) matters.

---

## Change 1 — remove the `robustness` axis (4-axis ObjectiveVector)

### 1a. `quant_fund_agent/research_eval/fitness.py`

- `ObjectiveVector` (line ~52): delete the `robustness` field and remove
  `"robustness"` from `AXES`. Change 4 then adds `temporal_robustness` in its
  place, so the FINAL tuple is
  `("marginal_value", "independence", "temporal_robustness", "parsimony",
  "structural_novelty")` (5 axes).
  `from_dict` already ignores unknown keys (it reads only `cls.AXES`), so old
  checkpoints/state files load fine — add a test asserting that.
- Update the module docstring (renumber the axes; state that plateau /
  perturbation / sign now fold into `marginal_value`).

### 1b. `quant_fund_agent/research_eval/harness.py`

- **Delete the PSR machinery**: `_prediction_pnl` (~line 300),
  `_marginal_pnl_series` (~317), `_psr_against_zero` (~345),
  `_empty_robustness` (~670). First `grep` each symbol repo-wide to confirm
  nothing else uses them.
- **Reshape `_robustness` (~570) into `_marginal_penalties`**: keep exactly the
  plateau-penalty block (jitter ICs on VAL), the perturbation-fidelity block,
  and the sign-consistency block; remove the PSR computation and the
  `with_pred`/`base_pred` parameters. Return
  `{"plateau_penalty", "jitter_ics", "perturbation_penalty",
  "sign_consistency"}`.
- **In `evaluate_candidate` (~876)**: the `marginal_value` axis becomes

  ```
  marginal_axis = marginal_value_raw
                  − plateau_weight · plateau_penalty        (if measured)
                  − perturbation_weight · perturbation_penalty (if measured)
                  + sign_bonus  (if sign_consistency is True)
                  − sign_bonus  (if sign_consistency is False)
  ```

  where `marginal_value_raw` is the LOCO ΔIC exactly as computed today.
  None-safety: if `marginal_value_raw` is None the axis stays None (penalties
  are not applied to nothing). Store `marginal_value_raw` in diagnostics.
  Remove every `robustness_*` diagnostics key; keep `plateau_penalty`,
  `jitter_ics`, `perturbation_penalty`, `sign_consistency` in diagnostics.
- **In `evaluate_set` (~1076)**: identical treatment for the SET-mode combined
  axis (its primary axis is the combined VAL IC; the combined-PSR robustness is
  removed; plateau/perturbation/sign fold onto the primary axis the same way).
- **`EvalParams`**: change the default `sign_bonus: float = 0.02` →
  **`0.002`**. Rationale (do not skip this): the bonus used to sit on a [0,1]
  PSR scale where 0.02 was small; on the marginal-ΔIC scale typical values are
  0.005–0.02, so an unchanged 0.02 bonus would dominate the axis. Keep
  `plateau_weight = 1.0` (the plateau penalty is already in IC units —
  coherent now). Keep `robustness_min_obs`, `lambda_std`, `stability_blocks`
  as fields with a "retained for API/CLI compatibility, unused" comment (the
  file already follows this convention for `lambda_std`/`stability_blocks`).
  Update the module docstring's "Family 4" section.

### 1c. `quant_fund_agent/agents/factor_research/evolution/reflection.py`

- Delete the PSR paragraph (lines ~91–101). Replace with one line reporting the
  primary-axis composition when penalties were applied, e.g.:
  `Marginal axis penalties: raw ΔIC {raw} → axis {axis} (plateau −{p},
  perturbation −{q}, sign bonus {±b}); OOS/IS degradation ratio {r}.`
  (The degradation ratio was reported inside the deleted paragraph — keep it
  reported.) All existing advice rules stay.

### 1d. Sweep for references

- `run_factor_evolution.py` ~line 201: the `--perturbation-weight` help text
  says "robustness axis" — reword to "marginal-axis perturbation probe".
- Grep the whole repo for `robustness` and `objective.robustness` and fix any
  remaining selection-channel references (checked at plan time: `publish.py`,
  `qd.py`, `controller.py`, `curation.py` do not touch the axis; tests do).

### 1e. Tests

- Update `tests/test_research_eval_fitness.py` (if present; else wherever
  `ObjectiveVector` is tested): AXES length 4; dominance over 4 axes;
  `from_dict` with a legacy dict containing `"robustness"` loads and ignores it.
- Update `tests/test_research_eval_harness.py`: penalties now move
  `marginal_value` (assert axis == raw − plateau, etc. on the synthetic panel);
  `robustness` axis absent; PSR helpers gone.
- Update every evolution test referencing the robustness axis
  (`tests/test_evolution_{controller,mutation,loop,debate,set_walkforward}.py`).

---

## Change 2 — residual IC scored on IS∪VAL

- `quant_fund_agent/research_eval/harness.py` `evaluate_candidate` (~line 922):

  ```python
  residual_ic = _residual_ic(
      candidate_signal, book, panel, cfg,
      split.is_mask,          # betas still fit on IS only (unchanged)
      split.is_val_mask,      # ← score on IS∪VAL (was: split.val_mask)
      split.is_val_mask,
  )
  ```

  Only the `score_mask` argument changes. `_residual_ic` itself is untouched
  (its `available_mask` logic already prevents forward-return labels crossing
  the dev boundary). Update its docstring and the Family-2/3 module docstring.
- Do **not** change the windows of standalone IC, `ic_decay`, jitter ICs, or
  the degradation gate (`is_ic` vs `val_ic` must stay era-separated — that gate
  is the era-consistency check and is deliberately preserved).
- SET mode: if `evaluate_set` computes a residual-IC-based quantity, apply the
  same score-window change; if it only uses internal PR/combined IC, no change.
- Test (`tests/test_research_eval_harness.py`): synthetic candidate whose edge
  exists only in the IS era (predictive on IS bars, pure noise on VAL bars) →
  under the new window the residual IC / independence axis is materially
  non-zero, whereas the old VAL-only value was ~0. Also assert TEST rows still
  never contribute (existing leak tests must stay green).

---

## Change 3 — progressive data reveal (flag-gated, default OFF)

### Semantics (fixed; do not redesign)

Let the run's panel index (after any `cutoff_date` slicing) have `n` bars.

- `test_start = floor(n · (1 − test_frac))`; **dev** = bars `[0, test_start)`.
  The tail `[test_start, n)` is the final TEST — never revealed, never scored
  in-run (exactly today's TEST semantics).
- `seed_end = floor(dev_len · seed_frac)` — the window visible at generation 0
  (seeding) and until the first reveal.
- Reveal generations: every generation `g ∈ {1..G}` with
  `(g − 1) % reveal_every == 0` is a reveal generation (so generation 1 always
  reveals). `R` = number of reveal generations. The remaining
  `dev_len − seed_end` bars are split into `R` equal blocks (remainder bars go
  to the **last** block) so the final reveal makes the whole dev window
  visible.
- At any generation, `visible_end` = seed_end + (blocks revealed so far) ·
  block_len (last block absorbs the remainder). **Expanding window**: blocks
  are never dropped. (A `max_window_bars` rolling cap was explicitly rejected —
  do not add one.)
- **Sliding VAL**: `val_span = val_blocks · block_len` bars;
  `val_start = visible_end − val_span`; IS = `[0, val_start)`,
  VAL = `[val_start, visible_end)`. Clamp: IS must keep ≥ 30 bars (the
  harness's minimum fit support); if the clamp binds, shrink `val_span`.
  At generation 0 the same formula applies with `visible_end = seed_end`.
- Validation at config time (raise `ValueError`): `0 < test_frac < 1`,
  `0 < seed_frac < 1`, `reveal_every ≥ 1`, `val_blocks ≥ 1`, `block_len ≥ 1`,
  and the generation-0 IS ≥ 30 bars.
- Purging at the frontier needs no new code: the harness's
  `available_mask`/`_label_available_mask` pattern already drops rows whose
  `t+h` label would cross the end of IS∪VAL, and the service slices the panel
  to the dev window so later bars are physically absent (see 3c).

### 3a. New module `quant_fund_agent/agents/factor_research/evolution/progressive.py`

Pure, panel-free, unit-testable:

```python
@dataclass(frozen=True)
class GenerationWindow:
    generation: int
    visible_end: int          # bar index (exclusive)
    val_start: int            # bar index
    is_end_ts: str            # ISO timestamp = index[val_start]
    val_end_ts: str           # ISO timestamp = index[visible_end]
    reveal: bool              # True if this generation revealed a new block
    block_bounds: tuple[str, str] | None   # [start_ts, end_ts) of the newly
                                           # revealed block (for prequential)

def build_schedule(index, *, generations, test_frac, seed_frac,
                   reveal_every, val_blocks) -> list[GenerationWindow]:
    ...  # implements the semantics above; index is the panel DatetimeIndex
```

Note the timestamp convention: `three_way_split` calendar mode takes IS =
`idx < is_end`, VAL = `[is_end, val_end)` — so `is_end_ts = index[val_start]`
and `val_end_ts = index[visible_end]` (use `index[visible_end]` directly when
`visible_end < n`; it always is, since `visible_end ≤ test_start < n`).

### 3b. Config + CLI

- `EvolutionRunConfig` (`evolution/loop.py` ~line 108) new fields, defaults
  chosen so OFF is byte-identical to today:

  ```python
  progressive: bool = False
  test_frac: float = 0.2       # final never-revealed tail (progressive mode)
  seed_frac: float = 0.45      # fraction of DEV visible at generation 0
  reveal_every: int = 1        # generations between reveals
  val_blocks: int = 2          # sliding VAL = last val_blocks blocks
  ```

- `run_factor_evolution.py`: `--progressive-reveal` (store_true),
  `--test-frac`, `--seed-frac`, `--reveal-every`, `--val-blocks`; thread into
  the config next to `is_frac`/`val_frac` (~lines 306–307). In progressive
  mode `is_frac`/`val_frac` are ignored (log a line saying so).
- Composes with `cutoff_date` (the outer walk-forward): the schedule is built
  on the **cutoff-sliced** index, which is what the service already evaluates
  on. No walk-forward code changes.

### 3c. MCP seam: calendar-mode split threading

Add optional `is_end: str | None = None, val_end: str | None = None` kwargs to:

- `quant_fund_agent/mcp/research_client.py` `evaluate_fitness` (~90) and
  `evaluate_set_fitness` (~137) — pass through.
- `quant_fund_agent/mcp/research_server.py` tool wrappers (~95, ~132).
- `quant_fund_agent/mcp/research_service.py` `evaluate_fitness` (~558) and
  `evaluate_set_fitness`: when both are provided, build
  `full_split = three_way_split(panel["close"].index, is_end=is_end,
  val_end=val_end)` instead of the fraction form (`splits.py` already supports
  calendar mode, ~line 104). Everything downstream is unchanged: `dev_mask =
  full_split.is_val_mask`, the panel is sliced to dev (so unrevealed bars are
  physically absent — TEST-invisibility extends to unrevealed data for free),
  and the harness receives the re-based split exactly as today.
- **Signal-cache correctness (critical):** `_cached_signal`
  (`research_service.py` ~508) is keyed by `(panel_key, cutoff_date, code
  fingerprint)` but computes the signal on the dev-sliced panel, whose length
  now varies per generation. A signal cached at an earlier (shorter) window
  must NOT be reused for a later (longer) one. Extend the cache key with
  `val_end` (the dev frontier; `None` in legacy mode). The cache stays bounded:
  one entry per (program, window) and there are only `R+1` windows per run.
- Add a small client+service helper `panel_timeline(data_dir, n_tickers,
  fields, cutoff_date) -> {"index": [iso strings]}` returning the cached
  panel's index (check first whether an equivalent metadata call already
  exists; reuse it if so). The loop calls it once at startup to build the
  schedule.

### 3d. Loop integration (`evolution/loop.py`)

- At `run()` start (before generation 0, ~line 1041): if `cfg.progressive`,
  fetch the timeline, `self._schedule = build_schedule(...)`, set
  `self._window = self._schedule[0]`.
- `evaluate_program` (~614) and `evaluate_set` (~651): pass
  `is_end=self._window.is_end_ts, val_end=self._window.val_end_ts` when
  progressive, else omit (None) — legacy behavior byte-identical.
- In the generation loop (~1095), for a reveal generation `g` do, **in this
  order, before proposing children**:
  1. **Prequential OOS score** (before the frontier moves): if the archive is
     non-empty, call `research_client.score_book_oos(book=<archive programs>,
     start=<old frontier ts>, end=<new frontier ts>, target_horizon,
     data_dir, n_tickers, fields, marginal_model)` — the window
     `[old, new)` was never visible to any selection that produced this
     archive, so this is honest OOS. Append one JSON line to
     `<out_dir>/prequential.jsonl`: `{generation, start, end, combined_oos_ic,
     n_obs, per_factor_ic, pbo, archive_size, n_trials}`. Empty archive →
     append a `{"skipped": "empty archive"}` row. (`score_book_oos` fits on
     all bars `< start` = exactly the visible window; both bounds are ≤ the
     dev end so the final TEST tail is untouched.)
  2. **Advance the frontier**: `self._window = self._schedule[g]`.
  3. **Re-score the archive on the new window**: for every archive member,
     re-run the evaluation path (same code as `evaluate_program` /
     `evaluate_set`: LOCO book = fixed_book + archive-minus-self as of the
     archive composition *before* re-scoring — snapshot the member list first
     so re-scoring is order-independent; regenerate jitter probes via
     `jitter_variants`, deterministic). Pass `n_trials=self.controller.
     n_trials` and do **NOT** call `controller.next_trial()` — re-scores are
     not new trials. Replace each member's `fitness`, then rebuild the archive
     as the gate-passing non-dominated set of the re-scored members (add a
     `controller.rescore_archive(new_fitness_by_genome_id)` method that
     replaces fitnesses and re-prunes in one pass). Members that drop out
     remain in `kept_pool` (they already are; kept_pool is not re-scored —
     end-of-run curation refits on the final window anyway). The QD grid is
     deliberately NOT re-scored (it is a diversity library, not the marginal
     reference) — leave its stored elites; document this in the module
     docstring. Append one lineage row per re-scored member:
     `{"event": "rescore", "generation": g, "genome_id", "objective_before",
     "objective_after", "gates_after"}` — the before/after drift is itself an
     overfitting diagnostic the thesis will plot.
  4. **Dedup retry**: call the new `controller.release_failed_fingerprints()`
     (see 3e).
- Stamp diagnostics: after every successful evaluation in progressive mode,
  set `fitness.diagnostics["scored_through"] = self._window.val_end_ts`,
  `["window_generation"] = self._window.generation`, and
  `["reveal_generation"] = self._window.reveal` (loop-side, after
  `FitnessResult.from_dict`). The reveal flag matters for Change 4: at reveal
  generations part of VAL is newly revealed, so the temporal-robustness values
  measured there are the ones computed on genuinely unseen data — post-run
  analysis must be able to separate them. Reflection: in `mutation_brief`, if
  `scored_through` is present, append one line
  `Scored on data through {date} (progressive reveal).`

### 3e. Controller additions (`evolution/controller.py`)

- Track gate-failers for retry: in `insert` (~240), if
  `not evaluated.fitness.gates.passed`, record the fingerprint in a new
  `self._failed_fingerprints: dict[str, int]` (fingerprint → times released).
- `release_failed_fingerprints(self, max_retries: int = 1) -> int`: for each
  failed fingerprint with count < max_retries, `discard` it from
  `self._fingerprints` and increment its count; return how many were released.
  (Rationale: on new data a previously gate-failing genome may honestly pass;
  cap = 1 retry per fingerprint so identical genomes can't be re-tried every
  reveal forever.)
- `rescore_archive(...)` as described in 3d step 3.
- Persistence: add `failed_fingerprints` to `to_state`/`save` (~422) and
  restore in `load` (~433–462). The loop's progressive state needs no extra
  persistence: the schedule is a pure function of (config, index) and the
  controller already persists `generation`, so resume recomputes
  `self._window = self._schedule[controller.generation]`. Assert on resume
  that the recomputed frontier matches a `frontier_ts` field written into
  `state.json`'s run metadata (cheap corruption guard).

### 3f. Tests — new `tests/test_evolution_progressive.py` (+ updates)

Follow the stub-LLM / synthetic-panel patterns already used in
`tests/test_evolution_loop.py`.

1. **Schedule math** (no panel needed beyond a synthetic DatetimeIndex):
   seed/block boundaries; last reveal reaches exactly the dev end; the TEST
   tail is never inside any window; `val_span` and the IS≥30 clamp; remainder
   bars land in the last block; determinism (same inputs → same schedule);
   config validation errors.
2. **Seam threading** (in-process, `QF_USE_MCP=0`): `evaluate_fitness` with
   `is_end`/`val_end` produces a split whose IS/VAL sizes match the calendar
   bounds; **signal-cache windowing** — evaluate the same program at two
   frontiers and assert the second evaluation's signal covers the longer
   window (no stale short-window reuse).
3. **Loop integration**: run a small progressive loop (2–3 generations) and
   assert: the frontier advances on schedule; archive members' fitness objects
   change on reveal generations while `n_trials` is unchanged by re-scoring;
   `prequential.jsonl` rows exist with `[start, end)` equal to the revealed
   block bounds and `end` ≤ the dev boundary; a gate-failing genome's
   fingerprint is re-evaluable after a reveal but only once.
4. **OFF-mode invariance**: with `progressive=False` the client is called with
   `is_end=None, val_end=None` and the whole existing test suite passes
   unchanged.
5. **Resume determinism**: run to generation 2, save, load, continue → final
   `state.json` equals the uninterrupted run's.

---

## Change 4 — temporal-degradation gate → fifth Pareto axis (added 2026-07-16)

This change is small — the ratio is already computed as the
`degradation_ratio` diagnostic, so the work is moving one number from the gate
channel to the objective vector — but the touch points are spread across
several files. Work through the whole list; do not stop after the harness.

### 4a. `research_eval/fitness.py`

- `ObjectiveVector`: add `temporal_robustness: float | None = None` in the
  position the deleted `robustness` field occupied → final
  `AXES = ("marginal_value", "independence", "temporal_robustness",
  "parsimony", "structural_novelty")`. The `None → −inf` dominance handling
  applies unchanged.
- `GateResults`: delete the `degradation_ok` field and remove it from `GATES`
  → `("coverage_ok", "deflation_ok", "cost_ok")`. `from_dict` reads only
  `cls.GATES` keys, so legacy state files containing `degradation_ok` still
  load — add a test.
- Update the module docstring (axis list + gate list).

### 4b. `research_eval/harness.py`

- `evaluate_candidate`: keep computing the degradation ratio exactly as today
  (`deg_ratio = (val_ic · sign(is_ic)) / |is_ic|`, not evaluable when
  `is_ic`/`val_ic` is missing or `|is_ic| < params.min_is_ic`), but instead of
  a gate:

  ```python
  temporal_robustness = (None if deg_ratio is None
                         else float(np.clip(deg_ratio, -1.0, 1.0)))
  ```

  Set it on the objective vector; delete the degradation gate block and the
  `reasons["degradation"]` entry; keep the RAW unclipped ratio in diagnostics
  as `degradation_ratio`.
  **Why the clip (do not skip it):** Pareto dominance uses only per-axis
  ordering, but crowding distance normalises by axis range — an uncapped
  ratio explodes when `|is_ic|` sits just above the evaluability floor
  (val 0.03 / is 0.006 = 5) and a single outlier flattens crowding for every
  other candidate; capping at 1 also removes the incentive to game
  tiny-denominator ratios (full retention of the in-sample edge is as good as
  it gets), and the floor at −1 bounds the sign-reversal side symmetrically.
- `evaluate_set` (~lines 1178–1190): identical conversion for the combined
  signal's degradation gate → the SET-mode objective's `temporal_robustness`.
- `EvalParams`: `gate_degradation` becomes unused — keep the field with the
  existing "retained for API/CLI compatibility, unused" comment convention.
  `min_is_ic` stays live (it now governs axis evaluability, not a gate) —
  update its comment.

### 4c. `agents/factor_research/evolution/reflection.py`

- Give the axis its own brief line; do NOT leave the degradation ratio inside
  the marginal-penalties line added by Change 1c. Suggested rendering:
  `Temporal robustness (axis): retained {ratio} of the in-sample edge on VAL
  (IS {is_ic}, VAL {val_ic}; axis value {clipped}).` with the usual n/a
  handling when unmeasured.
- The existing advice rule (`degradation_ratio < 0.5` → "classic overfit
  signature — simplify") stays unchanged. Verdict lines no longer mention a
  degradation gate, because it no longer exists.

### 4d. Reference sweep

- `grep -rn "degradation_ok"` repo-wide and fix every reference (tests,
  showcase/landing-examples, publish, QD — wherever it appears).
- `grep -rn "gate_degradation"` across `run_factor_evolution.py`, `loop.py`,
  `mcp/research_{client,server,service}.py`: remove any CLI flag wiring; keep
  the `EvalParams` field per 4b.

### 4e. Tests

- `fitness`: `AXES` == 5 including `temporal_robustness`; `GATES` == 3;
  legacy dicts containing `robustness` / `degradation_ok` keys round-trip.
- `harness` (synthetic panel): a candidate with a clear IS edge and an
  opposite-sign VAL edge now PASSES the gates and carries
  `temporal_robustness < 0`; `|is_ic| < min_is_ic` → axis `None` while the
  gates are unaffected; a ratio > 1 → axis exactly `1.0`; diagnostics keep the
  raw unclipped ratio.
- Update every existing test that asserts a degradation-gate failure or
  reads `degradation_ok`.

---

## Docs (required, project convention)

- `README.md` and `claude.md`: update the evolutionary-researcher status
  paragraphs — the CORE vector is now **5 axes** (PSR robustness axis removed,
  plateau/perturbation/sign folded into the marginal axis; the
  temporal-degradation hard gate became the clipped `temporal_robustness`
  axis, so the search gates are down to coverage + optional cost), residual IC
  scores on IS∪VAL, and progressive reveal exists behind
  `--progressive-reveal` (expanding window, sliding VAL, per-reveal archive
  re-scoring, prequential OOS log, final `test_frac` tail untouched).
- `docs/research-evolution/DESIGN.md`: update the objective-vector section the
  same way. `docs/factor_research_math_summary.tex`: update the axis list if it
  documents the five axes.
- `docs/research-evolution/IMPLEMENTATION_PROGRESS.md`: add a dated entry.
- `research_docs/SOURCES.md` (author's standing rule: log every borrowed
  external idea when implemented): add rows for the **prequential principle**
  (Dawid 1984, "Present position and potential developments: some personal
  views" — progressive reveal = prequential evaluation of the evolving book)
  and **dynamic training-subset selection in GP** (Gathercole & Ross 1994 —
  moving evaluation windows during evolution). Do NOT log Dwork 2015 /
  Blum–Hardt 2015 (reusable holdout / Ladder) — discussed but not implemented.

---

## Out of scope — do NOT implement

- Thresholdout / select-guard split (`holdout_ok` gate) — deferred; may be
  layered on later if the prequential curve shows old-prefix ratcheting.
- ε-dominance / Ladder admission thresholds; reflection-brief coarsening.
- Block-median aggregation of the marginal axis; CPCV cross-fitted marginal.
- Dev-wide scoring for standalone IC / `ic_decay` / jitter ICs (residual IC
  only, per Change 2).
- A rolling `max_window_bars` cap (explicitly rejected: expanding window).
- Progressive reveal for the GP arm (`agents/factor_research/gp/`) — the new
  kwargs default to None so the GP loop keeps working unchanged; do not modify
  `gp/` .
- Notebook updates.

---

## Suggested order & acceptance criteria

Stages (each ends with a full green `./venv/bin/pytest` and its own commit):

1. Change 1 (PSR axis removal + penalty fold + reflection + test updates).
2. Change 4 (degradation gate → `temporal_robustness` axis; pairs naturally
   with Change 1 since it touches the same files).
3. Change 2 (residual-IC window + test).
4. Change 3a–3c (schedule module + config/CLI + MCP seam incl. signal-cache
   key) + tests 1–2.
5. Change 3d–3e (loop/controller integration, prequential log, dedup retry,
   resume, reveal-generation stamp) + tests 3–5.
6. Docs.

Acceptance checklist:

- [ ] `ObjectiveVector.AXES` has 5 entries (`temporal_robustness` in place of
      `robustness`); `GateResults.GATES` has 3; legacy state files containing
      `robustness` / `degradation_ok` keys still load.
- [ ] A candidate with a sign-flipped VAL edge passes the gates and carries a
      negative `temporal_robustness`; a ratio > 1 clips to exactly 1.0; the
      raw ratio survives as the `degradation_ratio` diagnostic.
- [ ] `grep -rn "robustness" quant_fund_agent/` shows only the
      `temporal_robustness` axis, compat comments and unrelated factor-code
      comments — no PSR-based selection-channel references.
- [ ] Marginal axis = raw ΔIC − plateau − perturbation ± sign_bonus (default
      sign_bonus now 0.002), verified by tests.
- [ ] Residual IC scored on IS∪VAL, betas fit on IS, no TEST leak.
- [ ] `--progressive-reveal` off → behavior byte-identical (full old suite
      green, `is_end`/`val_end` = None everywhere).
- [ ] Progressive on → frontier schedule correct, archive re-scored per reveal
      without n_trials billing, `prequential.jsonl` written, signal cache
      window-keyed, resume deterministic, final `test_frac` tail never read.
- [ ] README / claude.md / DESIGN.md / IMPLEMENTATION_PROGRESS.md / SOURCES.md
      updated.
