# Joint Factor×Execution Evolution — Outer-Layer Design & Architecture

Updated: 2026-07-11
Status: **AGREED — ready to build (nothing implemented yet).** All design
forks were resolved with the author on 2026-07-11 (see §Decision record).
This document is the implementation-anchor *prompt* for the outer layer —
the same role `docs/research-evolution/DESIGN.md` and
`docs/execution-evolution/DESIGN.md` play for the two inner loops; a
companion `IMPLEMENTATION_PROGRESS.md` will track the phases once the build
starts. The two inner loops are specified elsewhere and are **consumed, not
re-specified, here**: inner loop 1 = the built evolutionary factor
researcher (`docs/research-evolution/DESIGN.md`), inner loop 2 = the
execution researcher (`docs/execution-evolution/DESIGN.md`, revised
2026-07-11 into block form).

## Purpose

The factor researcher evolves alpha programs; the execution researcher (once
built) evolves the programs that trade the alpha. Run one after the other —
the original plan — the pipeline is *greedy per layer*: the factor search
never learns net-of-cost tradability (a factor with strong gross IC but a
turnover profile no executor can trade cheaply survives stage 1, and stage 2
cannot fix it), and the execution search co-adapts to whatever book stage 1
happened to finish with.

This document designs the **thin outer layer** that turns the two loops into
a *joint* optimisation without merging them: a **block-coordinate
alternation** in which each arm evolves for a *block* (G generations) against
the other arm's **frozen SOTA state**, with a scheduler (sequential /
round-robin / random / bandit) deciding which arm gets the next block, a
**re-freeze protocol** at block boundaries, and a **shared cross-arm
`N_trials` ledger** keeping the multiple-testing accounting honest across
the whole run. The sequential two-stage pipeline, round-robin alternation and
bandit-scheduled joint search are all *one scheduler config* of the same
runner — the thesis ablation matrix falls out of one codebase.

Two deliberate exclusions (locked): **the predictive model is not an arm** —
the combined model stays a deterministic refit inside each arm's evaluation
harness (the factor arm's marginal-value combiner; the frozen-signal fits),
in deliberate contrast to RD-Agent(Q), whose second arm is model
configuration (config-AutoML — the least defensible contribution class, as
already argued in `docs/execution-evolution/DESIGN.md` §Why program
synthesis). And **no LLM anywhere in the outer layer** — the scheduler, the
re-freeze, and the joint objective are all deterministic; LLM agency lives
entirely inside the arms' mutation operators, so the prime directive (the
LLM never influences its own reward) holds at every level of the stack.

## The single most important principle (unchanged, now three layers deep)

> **The LLM ideates and mutates; a deterministic harness scores. The LLM
> must never be able to influence its own reward.**

The outer layer adds a new place this could silently break: the *scheduler*.
An LLM-chosen schedule (RD-Agent(Q) actually ablates one) would let language
output steer which hypotheses get tested against VAL. We therefore restrict
schedulers to deterministic or classical-stochastic policies (round-robin,
random, Thompson sampling over *numeric* block rewards) — the same reason
the reflection brief is rule-based.

## Decision record (2026-07-11, resolved with the author)

1. **Layered architecture** — build execution evolution per its own DESIGN
   (inner loop 2), then add this thin outer layer. Not two forever-separate
   stages; not a literal single-population merge.
2. **Cross-layer coupling = additive, default-off** — when a SOTA executor
   exists, the factor arm's existing `cost_ok` gate upgrades to
   net-of-cost-*through-the-SOTA-executor*, plus a `net_capture_sota`
   diagnostic in the reflection brief. The 5-axis Pareto vector is untouched;
   with coupling off, every baseline arm is byte-identical to a standalone
   run (test-enforced).
3. **Scheduler: round-robin default; contextual Thompson-sampling bandit as
   `--scheduler bandit`**; `sequential` (the two-stage plan) and `random`
   (RD-Agent(Q)'s ablation control) complete the set.
4. **Two arms only** — factor programs and execution programs.

## RD-Agent(Q): what we take, what we replace

**The paper.** Li et al., *R&D-Agent-Quant: A Multi-Agent Framework for
Data-Centric Factors and Model Joint Optimization* (NeurIPS 2025,
arXiv:2505.15155; code: microsoft/RD-Agent). The load-bearing reading: it is
**not one population**. It maintains a SOTA factor library and a SOTA model;
a **contextual Thompson-sampling bandit** — arms `{factor, model}`, 8-dim
context `[IC, ICIR, RankIC, RankICIR, ARR, IR, −MDD, SR]`, one Bayesian
linear regression per arm, reward = performance improvement of the iteration
— picks which arm gets the next R&D iteration; factor iterations are
evaluated **against the frozen SOTA model** and model iterations against the
frozen SOTA factor set. "Joint optimization" = alternation + shared SOTA
state + adaptive budget allocation. Their scheduler ablation motivates ours:
bandit IC 0.0532 / ARR 0.1421 vs LLM-chosen 0.0476 / 0.1009 vs random
0.0445 / 0.0897.

**What we take:** block-coordinate alternation over shared SOTA state;
frozen-counterpart cross-evaluation; the contextual-TS scheduler (as an
ablation arm, not the default); the scheduler-ablation experiment design
(bandit vs round-robin vs random).

**What we replace (the thesis delta):**

* **Statistics.** RD-Agent(Q)'s overfitting control is a redundancy filter
  (dedup at `IC_max ≥ 0.99`), a single train/val/test split, and
  median-of-5-seeds. No deflation, no PBO, no CPCV, no multiple-testing
  accounting — and their bandit *conditions on validation performance* with
  nothing pricing that adaptivity. We bring the full apparatus of the factor
  build (N_trials-aware deflation, PBO/CSCV, CPCV, dev-slice leak-freedom)
  plus two things that only exist *because* of the outer layer: a **shared
  cross-arm trials ledger** (§Shared N_trials ledger) and a **joint
  walk-forward** in which the entire outer loop re-runs per fold so the
  headline numbers never touch the search's VAL (§VAL re-burning).
* **The second arm.** Theirs is model configuration; ours is **execution
  programs** — path-dependent signal→book code synthesis with cost-aware,
  causality-probed scoring. A materially harder and less-explored layer.
* **Archives, not single-SOTA replacement.** Each of our arms keeps a
  Pareto/QD archive (a *book* of complementary factors; a *library* of
  execution skills), where RD-Agent(Q) keeps one SOTA object per arm. The
  SOTA state we exchange between arms is a *view* (curated book; best
  executor) over those archives, not the archive itself.

In the classical vocabulary this is **cooperative coevolution** (Potter & De
Jong): two populations, each evaluated against a frozen *representative* of
the other — here representative = SOTA state, generations = blocks — with
modern multiple-testing control layered on top. Naming the frame matters for
the paper: it connects a 2025 agents-paper mechanism to a 25-year-old
evolutionary-computation literature with known failure modes (relative
overgeneralisation, representative staleness), both of which our re-score
protocol and coupling design address explicitly (§Risks).

## SOTA state + persistence

```python
@dataclass
class SOTAState:                       # joint_evolution/state.py
    book: list[BookEntry]              # curated factor book: id, code, metadata (the factor arm's publish view)
    book_hash: str                     # dedup/provenance fingerprint of the book
    sota_executor: dict | None         # executor_id, code, params, fitness snapshot, provenance; None until the first exec block
    frozen_signals_version: int        # k of frozen_signals/v<k>/
    frozen_signals_manifest: str       # path to the current FrozenSignalSet manifest
    block_index: int                   # how many blocks have completed
```

`SOTAState.save/load` round-trips JSON. Everything persists under the
workspace `Scope` seam — **one joint run = one prerun**, exactly like factor
and GP runs (a new additive `Scope.joint_dir` property; nothing else in
`workspace.py` changes):

```
data/workspaces/<config>/preruns/<name>/joint/
  joint_state.json          # SOTAState + scheduler posterior + TrialsLedger
  blocks.jsonl              # one row per block (the outer lineage): arm, generations,
                            #   J_before/J_after, reward, ledger deltas, context vector,
                            #   sampled θ (bandit), refreeze version bump
  frozen_signals/v<k>/      # manifest.json + signal_<i>.parquet + poison_audit.json
  factor/                   # the factor arm's out_dir (state.json, lineage.jsonl, run_config.json)
  exec/                     # the exec arm's out_dir (same trio)
```

Checkpoint after **every block boundary** (state + scheduler + ledger +
both arm checkpoints are already on disk) → a joint run is resumable
mid-schedule exactly like an arm run is resumable mid-generation. Final
materialisation goes through the existing `persist_archive` path (factor
book) and the exec arm's persistence (E4 `executor_id` seam), so downstream
consumers (comparison harness, walk-forward backtest, PM) see a standard
prerun.

## Block protocol + re-freeze seams

A **block** = one incremental evolution session of one arm:
`joint_evolution/blocks.py::run_factor_block(...)` /
`run_exec_block(...)`. Each builds the arm's run-config via
`dataclasses.replace` over a base config (fixed `out_dir` under
`joint/factor/` or `joint/exec/`, ledger-fed family `n_trials`, coupling
injection when on), then calls the arm loop's block API —
`EvolutionLoop.run(resume=block_index > first_for_arm,
n_generations=G)` — and returns a `BlockResult` (generations run, candidates
scored, gate-pass counts, archive delta, wall time). The arm loops are
**unchanged code paths**; blocks are just resumed runs (the E1 seam of the
execution DESIGN, mirrored additively onto the factor loop).

**Boundary protocol** (deterministic, order fixed):

* **After a factor block** — `joint_evolution/refreeze.py::
  refreeze_after_factor_block(sota, controller, cfg)`:
  1. Curate/publish the arm's current kept-pool/archive into the new book
     view, reusing the existing `curate_book` / `publish_book` semantics
     (same curation mode the run was configured with; publish filter at the
     ledger's current joint count).
  2. Write the new book + `book_hash` into `SOTAState`.
  3. `execution/signal_freeze.freeze_eval_signals(book, ...)` → new
     `FrozenSignalSet` `v(k+1)` (IS-only fits, label-availability discipline,
     poison audit — see the execution DESIGN §Overfitting).
  4. `ExecEvolutionLoop.rescore_archive(frozen_v_k_plus_1)` — every archived
     executor deterministically re-scored against the new signals (fitness
     replaced in place, lineage annotated with the version bump). Re-scores
     bill **joint looks only** (§Ledger).
* **After an exec block** — `refreeze.py::update_sota_executor(sota, loop)`:
  `ExecEvolutionLoop.sota_executor()` (selection rule: gate-passing archive
  member with max primary axis; ties → cost efficiency → parsimony → lowest
  `genome_id`) written into `SOTAState`; the factor arm's coupling seam sees
  it at its next block. No factor-side re-scoring in v1: the coupling touches
  only the `cost_ok` gate and diagnostics, not the factor Pareto axes, so
  archived factor fitness does not go stale when the executor changes (the
  gate is re-evaluated for *new* candidates; a documented, deliberate
  asymmetry).

**The joint objective** `J` — `joint_evolution/objective.py::
score_joint_state(sota, split, params)` (new MCP service function
`score_joint_state`, same client→server→service pattern, in-process under
`QF_USE_MCP=0`): the **deflated net-of-cost VAL Sharpe of the full pipeline**
— combined curated-book signal (IS-fit combiner) → SOTA executor (baseline
seed executor while `sota_executor is None`) → the existing cost layer —
deflated at the ledger's current `n_joint_looks` via the existing
`deflated_sharpe_ratio`. Computed at every block boundary *after* the
re-freeze, so consecutive values are like-for-like; `ΔJ` across a block is
the scheduler's reward and `blocks.jsonl` records the trajectory.

## Scheduler design

`joint_evolution/scheduler.py`, one interface:
`pick(context, history) -> "factor" | "exec"` and
`update(arm, reward, context)`.

* **`SequentialScheduler`** — all factor blocks, then all exec blocks: the
  original two-stage plan as a degenerate schedule (the `sequential`
  experiment arm, and the standalone-equivalence regression anchor).
* **`RoundRobinScheduler`** *(default)* — deterministic alternation,
  factor first. Zero tuning, fully reproducible; the honest default until
  the bandit earns its keep in the ablation.
* **`RandomScheduler`** — uniform coin flip per block, seeded. RD-Agent(Q)'s
  ablation control, kept for the same reason.
* **`BanditScheduler`** — contextual Thompson sampling, the RD-Agent(Q)
  mechanism adapted:
  - **Arms:** `{factor, exec}`.
  - **Context** `x ∈ R^8`, this project's metrics (all deterministic,
    computed by `objective.py::context_vector(sota, history)`):
    `[combined-book VAL IC, CPCV mean/std of the book (ICIR analogue),
    effective-n-factors (participation ratio), 1 − max inter-factor |corr|
    (residual headroom), deflated net VAL Sharpe through the SOTA executor,
    net÷gross capture, turnover ÷ ceiling, EMA of this arm's past block
    rewards]`. Standardised online (running mean/std).
  - **Model:** one Bayesian linear regression per arm
    (`θ_a ~ N(μ_a, Σ_a)`, conjugate normal prior `μ=0, Σ = g·I` with heavy
    shrinkage `g` documented in config); Thompson step samples `θ̃_a` per
    arm, picks `argmax_a x'θ̃_a`; posterior update on the observed reward.
  - **Reward:** the block's **deflated `ΔJ`** (§Block protocol) —
    deflation keeps the reward from drifting upward merely because more
    looks accumulated.
  - **Cold start** — the honest problem at this granularity: a thesis run
    has ~5–15 block decisions, not RD-Agent's 30+ iterations. Mitigations:
    the first two blocks are forced round-robin (one observation per arm);
    the shrinkage prior keeps posteriors wide so exploration survives small
    n; `--bandit-context off` degenerates to non-contextual Gaussian TS on
    rewards alone (a simpler, lower-variance arm we also report); RNG seeded
    `run_seed + block_index` so schedules are exactly reproducible.
  - Posterior state serialises into `joint_state.json`.

Scheduler decisions are **not billed to the ledger** — the scheduler
reallocates which looks happen, it does not add hypotheses; its adaptivity
is priced by the joint walk-forward instead (§VAL re-burning, mitigation iv).

## Shared N_trials ledger

`joint_evolution/ledger.py::TrialsLedger` — `bill(arm, rescore=False)`,
`family_count(arm)`, `joint_count()`; serialised in `joint_state.json`.
Two levels, statistically distinct on purpose:

* **Per-family counts** (`n_factor`, `n_exec`) = unique candidates *scored*
  per arm across ALL blocks of the run (the existing billing rule — only
  scored candidates count — unchanged). These drive each arm's **own
  within-search deflation gates**: the `√(2·ln N)`-type haircut corrects the
  within-family maximum, and factor ICs vs executor Sharpes are different
  test families with different nulls — cross-billing them into each other's
  gates would be statistically wrong *and* would break byte-identity of the
  standalone baselines. Mechanically: the block injects `family_count(arm)`
  into the arm controller at block start and reads the incremented count
  back at block end (the controller already persists `n_trials` in its
  checkpoint, so this is free).
* **Joint look count** (`n_joint_looks`) = every harness evaluation that
  reads VAL, across both arms, **including archive re-scores after a
  re-freeze**. A re-score of an old candidate against new frozen signals is
  not a new hypothesis (it does not increment `n_exec`) — but it *is* a
  fresh look at VAL and one more draw in the implicit max over
  configurations the run performs, so it bills the joint count. By
  construction `n_joint_looks ≥ n_factor + n_exec`. The joint count is what
  the **final publish filter** and the **Statistician's touch-once TEST
  pass** deflate by — conservative where it must be: on the finally-selected
  configuration.

## VAL re-burning + the final validation protocol

Every block burns the same VAL window again, and the bandit *conditions on*
VAL-derived rewards — an adaptivity a static ledger alone cannot fully
price. Mitigations, in order of strength:

1. **Joint-count deflation** (above) — every VAL look is billed somewhere,
   and the final numbers are deflated at the total.
2. **The split is fixed once per outer run.** Blocks never re-split; both
   arms and every re-freeze see the identical IS/VAL/TEST masks — no window
   drift between arms, no accidental VAL growth.
3. **Frozen signals are always IS-fit** with the label-availability
   discipline, re-audited at every re-freeze (a leaky frozen signal would
   launder look-ahead into every executor score — the worst leak at this
   layer; see the execution DESIGN §Leak-free conventions).
4. **The headline protocol: the joint walk-forward.**
   `joint_evolution/walkforward.py::run_joint_walk_forward(base_cfg,
   boundaries, out_dir)` — for each fold `d_i`: a **fresh
   `JointEvolutionLoop`** (fresh ledger, fresh scheduler posterior, fresh
   arm states) runs the WHOLE outer loop — scheduler, blocks, re-freezes,
   both arms — with `cutoff_date=d_i` (the same cutoff seam the factor
   walk-forward threads today), then the final `(book, executor)` pair is
   scored **once** on `[d_i, d_{i+1})` via a new touch-once MCP call
   `score_joint_oos(book, executor, start, end, ...)` → combined OOS IC,
   net-of-cost Sharpe, net÷gross capture, turnover, MDD, plus PBO over the
   fold's executor-candidate P&L matrix. Validation only; nothing persisted
   to factor DBs — the exact analogue of
   `agents/factor_research/evolution/walkforward.py`, one level up. Inside a
   fold, VAL re-burning is then a *search-efficiency* concern, not a
   *validity* concern: the reported number never saw the scoring window.
5. VAL sub-rotation across blocks (different VAL sub-windows per block) is a
   **documented extension, not core** — it would break like-for-like
   comparability of block rewards for the bandit.

## Coupling: executor-aware factor fitness (additive, default-off)

Exactly one seam, threaded end-to-end:

* `research_eval/harness.py::EvalParams.cost_executor: dict | None = None`.
  When `None` (default): byte-identical behaviour, enforced by the named
  test `test_cost_executor_none_is_byte_identical`.
* When set (the SOTA executor's spec): `_turnover_netcost` builds the
  candidate's positions by running its signal **through the SOTA executor**
  (via the `execution/` package) instead of the explicit
  `zscore_over_time` + `directional_positions` construction; `cost_ok` then
  means *"net-of-cost return through the SOTA executor > 0 AND turnover ≤
  ceiling"*. A new `net_capture_sota` diagnostic (net÷gross through the
  executor) lands in `diagnostics`.
* `reflection.py` gains one deterministic advice rule: *"your factor loses
  X% of gross P&L through the current execution layer — consider slower /
  faster signal variants"* — the teacher channel's version of
  execution-awareness.
* Threading: `EvolutionRunConfig.sota_executor` →
  `research_client.evaluate_fitness(cost_executor=…)` /
  `evaluate_set_fitness(...)` → server → service → harness (the standard
  flat-kwargs pattern). `joint_evolution/blocks.py` injects
  `sota.sota_executor` when `--coupling on`.

The five Pareto axes are untouched; coupling changes a *gate* and the
*teacher channel* only. This is deliberate: it keeps every factor-arm run
comparable across the ablation matrix (same objective geometry), while still
letting execution-awareness shape *what survives* and *what the LLM is told*.
Promoting net-of-cost value to a primary axis is a documented extension.

## Experiment matrix (the paper's core table)

| Arm | Scheduler | Coupling | Claim under test |
| --- | --- | --- | --- |
| factor-evolution only (existing runs) | — | — | control 1: alpha search alone |
| exec-evolution only, baseline-book signals | — | — | control 2: execution search alone |
| sequential (factor → exec) | `sequential` | off | does staging beat the controls? |
| round-robin joint | `round_robin` | off | does alternation beat staging? |
| random-scheduler joint | `random` | off | RD-Agent(Q)'s ablation control |
| bandit joint | `bandit` | off | does adaptive allocation beat round-robin/random? |
| bandit joint + coupling | `bandit` | on | does cross-layer feedback help beyond allocation? |
| GP-factor-arm joint | `round_robin` | off | non-LLM factor arm: pluggability + LLM-value row |

All rows share panel/splits/costs/seeds. Headline metrics, **OOS-only via
the joint walk-forward**: net-of-cost deflated Sharpe (at `n_joint_looks`),
combined OOS IC, net÷gross capture, turnover, max drawdown, PBO. Per-run
descriptive plots from `blocks.jsonl`: the `J` trajectory over blocks, the
bandit's arm-choice history and per-arm reward posteriors (the qualitative
"where did the budget go" figure), re-freeze version timeline. Include the
per-arm compute/LLM-call budget per row so the allocation claim is
cost-honest.

## Related work positioning

* **FunSearch / AlphaEvolve** (DeepMind) — LLM-mutation + deterministic
  scorer: both inner arms.
* **RD-Agent(Q)** (Li et al., NeurIPS 2025) — the outer layer's anchor:
  block-coordinate alternation + contextual-TS scheduling. Our delta:
  statistics (ledger, deflation, joint walk-forward), the execution-program
  arm, archives instead of single-SOTA.
* **AlphaGen** (KDD 2023) — collection-level reward for factor mining; our
  LOCO marginal axis is the Pareto-native version.
* **AutoAlpha** (IJCAI 2020) — the GP benchmark arm's anchor.
* **EoH — Evolution of Heuristics** (2024) — LLM-driven heuristic evolution;
  closest general-domain relative of the arms' operator design.
* **FinMem / FinAgent** — runtime LLM agency; rejected (locked decision 8 of
  the execution DESIGN): our deployment is fully deterministic.
* **ADAS** (Hu et al., 2024) — meta-agent search over agent designs;
  orthogonal (could later tune the arms' prompts, not the trading logic).
* **Bailey & López de Prado** — deflated Sharpe (2014), PBO/CSCV (2016): the
  ledger's teeth; Harvey et al. (2016) for the multiple-testing framing.
* **Potter & De Jong** — cooperative coevolution (1994/2000): the classical
  frame for "two populations, each evaluated against the other's frozen
  representative". Our block+SOTA protocol is the sequential CCEA with
  representative = SOTA state, plus deflation the CCEA literature never had.
* **MAP-Elites / QD** (Mouret & Clune) — the archive structure inside each
  arm (already built on the factor side).

## Risks

* **Coupled non-stationarity** — each arm's archived fitness can go stale as
  the counterpart moves. Mitigated: exec archive is re-scored at every
  re-freeze (never discarded); factor-side staleness is structurally avoided
  in v1 by coupling only the gate/diagnostics, not the axes; coupling
  default-off keeps a clean-statistics path.
* **Bandit variance at n≈10 decisions** — mitigated by forced round-robin
  warmup, heavy-shrinkage priors, the non-contextual fallback, and
  round-robin as the *default* (the bandit must win the ablation to be the
  headline).
* **Re-freeze churn** — a much-changed book could invalidate executor
  progress. Mitigated: re-scored not discarded; `blocks.jsonl` records the
  archive-survival rate per re-freeze so churn is measurable; if it proves
  destructive, freeze-every-k-blocks is a config knob, not a redesign.
* **Ledger mis-implementation** — the family/joint split is subtle and
  load-bearing. Guarded by dedicated billing tests (a re-score must move
  `n_joint_looks` and not `n_exec`) and the sequential-equals-standalone
  byte-identity regression.
* **Compute** — the joint walk-forward multiplies a full outer run by the
  number of folds. Budget it explicitly in the run plan (generations-per-
  block × blocks × folds); the panel/signal caches and the cheap exec
  bar-loop keep per-candidate cost low; a reduced-budget walk-forward
  configuration (fewer blocks per fold) is acceptable *if identical across
  arms*.
* **Story sprawl** — the framework subsumes two papers' worth of machinery.
  The paper's claim discipline: the *arms* are prior work (ours); the
  *contribution under test* is the outer layer (allocation + coupling +
  statistics). The matrix is built so each row isolates one claim.

## Phasing (J-phases; interleaved with the execution arm's E-phases)

Agreed build order: **E0 → E1 → E2 → J0 → J1 → E4 → J3 → J2 → E3 → E5+J4**
(outer layer de-risked as soon as a jitter-only exec loop exists; E3
debate/RAG is the first cut if time compresses). Each phase ends green on
`./venv/bin/pytest` and updates `IMPLEMENTATION_PROGRESS.md`.

* **J0 — Outer state + ledger + sequential runner.**
  `quant_fund_agent/joint_evolution/{__init__,state,ledger,blocks,objective,
  loop}.py` + `run_joint_evolution.py` + additive `Scope.joint_dir`:
  `SOTAState.save/load`; `TrialsLedger.bill/family_count/joint_count`;
  `run_factor_block`/`run_exec_block` (family-count injection in/out);
  `score_joint_state` (+ MCP service/server/client);
  `JointRunConfig` (total_blocks, gens_per_block, scheduler="sequential",
  coupling=False, nested arm configs); `JointEvolutionLoop.run()` with
  per-block checkpoint + `blocks.jsonl`. Tests: `tests/test_joint_state.py`,
  `tests/test_joint_ledger.py` (family vs joint billing incl. rescore),
  `tests/test_joint_loop.py::test_sequential_equals_two_standalone_runs`
  (byte-identity of the factor arm vs a plain `run_factor_evolution` run,
  same seed — the key regression).
* **J1 — Round-robin + re-freeze protocol.** `scheduler.py` (Sequential /
  RoundRobin / Random behind the common `pick`/`update` interface) +
  `refreeze.py` (`refreeze_after_factor_block`, `update_sota_executor`,
  `rescore_exec_archive`). Tests: `tests/test_joint_scheduler.py`
  (per-seed determinism), `tests/test_joint_refreeze.py` (new signal version
  + poison audit; archive re-scored with `n_exec` unchanged and
  `n_joint_looks` billed; SOTA-executor tie-breaking).
* **J2 — Bandit scheduler.** `BanditScheduler` (contextual TS per
  §Scheduler; `context_vector` in `objective.py`; posterior serialised in
  `joint_state.json`); `--scheduler bandit`, `--bandit-context {on,off}`.
  Tests: converges to the better arm on synthetic stationary rewards within
  ~10 decisions; warmup forces one block per arm; posterior save/load
  round-trip; seeded determinism.
* **J3 — Coupling.** `EvalParams.cost_executor` through
  client/server/service/harness; `net_capture_sota` diagnostic + reflection
  rule; `EvolutionRunConfig.sota_executor`; `--coupling {off,on}` (blocks
  inject the SOTA executor). Tests:
  `test_cost_executor_none_is_byte_identical` (load-bearing) + executor-path
  correctness + a coupling-arm joint-loop smoke test.
* **J4 — Joint walk-forward + experiment matrix.**
  `joint_evolution/walkforward.py::run_joint_walk_forward` + touch-once MCP
  `score_joint_oos`; `--walk-forward d0,d1,…` on `run_joint_evolution.py`;
  the experiment matrix as documented CLI recipes (one command per row —
  no new harness). Tests: `tests/test_joint_walkforward.py` (two-fold
  synthetic run; fold isolation — fresh ledger/scheduler per fold;
  touch-once).

**Documentation convention** (inherited): after J2, a
`notebooks/joint_evolution_walkthrough.ipynb` (every component in isolation,
offline/synthetic, no API key: ledger billing, freeze/re-freeze round-trip,
one round-robin + one bandit block sequence with a stubbed evaluator) and a
`notebooks/joint_evolution_live_run.ipynb` (tiny end-to-end with real LLM
calls — 2 blocks × 2 generations — showing intermediate I/O). The execution
arm's own notebooks land after E2 (see its DESIGN).

**Changes the joint layer forces on the execution build NOW** (annotated in
the execution DESIGN, 2026-07-11 revision): `FrozenSignalSet` as a versioned
interface module in E0 (not run-start-inline); `run(resume, n_generations)`
on BOTH loops in E1; `sota_executor()` + `rescore_archive()` on the exec
loop in E1; executor fitness round-trips as the standard `FitnessResult`
dict; family-`n_trials` injectability (already free — the controller
persists `n_trials`).

## References

- Li et al. — *R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric
  Factors and Model Joint Optimization* (NeurIPS 2025, arXiv:2505.15155).
- DeepMind — *FunSearch* (Nature 2024); *AlphaEvolve* (2025).
- Bailey & López de Prado — *The Deflated Sharpe Ratio* (2014); *The
  Probability of Backtest Overfitting* (2016).
- Harvey, Liu & Zhu — *…and the Cross-Section of Expected Returns* (2016).
- Potter & De Jong — *Cooperative Coevolution: An Architecture for Evolving
  Coadapted Subcomponents* (Evolutionary Computation, 2000).
- Yu et al. — *AlphaGen* (KDD 2023). Zhang et al. — *AutoAlpha* (IJCAI 2020).
- Liu et al. — *Evolution of Heuristics (EoH)* (ICML 2024).
- Mouret & Clune — *Illuminating search spaces by mapping elites* (2015).
- Hu et al. — *ADAS: Automated Design of Agentic Systems* (2024) — orthogonal.
- Yu et al. — *FinMem / FinAgent* (2023–24) — runtime agency, rejected.
