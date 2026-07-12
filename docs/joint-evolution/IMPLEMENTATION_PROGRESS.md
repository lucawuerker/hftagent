# Joint-Evolution Implementation Progress (J-phases)

> **Handoff document** — keep current after every session. Design anchor:
> `docs/joint-evolution/DESIGN.md`. The execution arm's phases (E0–E5) are
> tracked in `docs/execution-evolution/IMPLEMENTATION_PROGRESS.md`.
> Run tests with `./venv/bin/python -m pytest`.

**Agreed build order:** E0 → E1 → E2 → J0 → J1 → E4 → J3 → J2 → E3 → E5+J4.

| Phase | Status | Notes |
| --- | --- | --- |
| J0 — outer state + ledger + sequential runner | **DONE 2026-07-12** | built together with J1 (sequential needed the freeze boundary anyway) |
| J1 — round-robin + re-freeze protocol | **DONE 2026-07-12** | `scheduler.py` (sequential/round_robin/random), `refreeze.py` (audit-gated re-freeze + exec-archive rescore) |
| J2 — bandit scheduler | **DONE 2026-07-12** | `joint_evolution/bandit.py::BanditScheduler` — per-arm Bayesian linear regression + TS; warmup forces one obs/arm; `context="on"` uses the 8-dim `context_vector(history)`, `"off"` = non-contextual Gaussian TS; posterior persists in joint_state.json; seeded per (seed, decision). `--bandit-context`. Converges to the better arm ≤14 decisions (min 5/6 tail over 20 seeds, both variants). Tests in `test_joint_scheduler.py` (9 total). GOTCHA fixed: Bayesian-ridge posterior mean must be `inv(Λ)·(Σx·r)/σ²` — an early version scaled it by σ² (arms indistinguishable). |
| J3 — coupling (`EvalParams.cost_executor`) | **DONE 2026-07-12** | `EvalParams.cost_executor` (None = byte-identical, test-enforced); `_turnover_netcost` builds positions THROUGH the SOTA executor (compile_executor_inmem, smoke=False; broken executor falls back + logs, never crashes the factor arm); `cost_ok` additionally requires net-of-cost-through-executor > 0; `net_capture_sota`/`cost_executor_id` diagnostics (only when coupling on); factor reflection gained the net-capture advice rule; threaded client→server→service (evaluate_fitness + evaluate_set_fitness) + `EvolutionRunConfig.sota_executor` + `blocks.run_factor_block` + `--coupling`. Tests appended to `test_research_eval_harness.py` (3: byte-identity, gate+diag, broken-executor fallback). |
| J4 — joint walk-forward + experiment matrix | **DONE 2026-07-12** | `joint_evolution/walkforward.py::run_joint_walk_forward` — fresh JointEvolutionLoop per fold (`cutoff_date=d_i`, own dir/ledger/posterior), touch-once `score_joint_oos` on `[d_i, d_{i+1})` (net/gross OOS Sharpe, composite OOS IC, capture, turnover, MDD); `run_joint_evolution.py --walk-forward d0,d1,…`; writes `walkforward.json` incrementally. Tests: `test_joint_walkforward.py` (2, incl. the poison touch-once check). The experiment MATRIX = CLI recipes in DESIGN §Experiments (runs are thesis work). PBO over the exec-candidate P&L matrix in the fold scorer = documented extension. |

## What exists (J0+J1, all tests green)

- `quant_fund_agent/joint_evolution/{__init__,state,ledger,scheduler,refreeze,blocks,objective,loop}.py`
- `run_joint_evolution.py` (Scope prerun; `--scheduler --total-blocks --gens-per-block --coupling` etc.)
- `workspace.py::Scope.joint_dir` (additive)
- MCP `score_joint_state` (service+server+client): J = per-bar VAL net Sharpe of
  book→combined-ridge→executor→costs, **minus the `√(2·ln n_looks)/√n_obs` haircut**
  (per-bar Sharpe null std ≈ 1/√n_obs so the IC haircut applies verbatim); DSR prob
  reported with `sr_variance = 1/n_obs`.
- Ledger: `bill(arm, n, rescore=)` (family+joint / joint-only) + `bill_look(n, source)`
  for block-boundary objective scores. Invariant: `n_joint_looks ≥ n_factor + n_exec`.
- Block accounting: the arm's own `controller.n_trials` IS its family count (each arm
  owns its out_dir); blocks bill the ledger with the per-block delta via
  `loop._n_trials_at_entry` (set in both arms' `run()`).
- Block 0 is ALWAYS factor (exec needs a frozen book); enforced in the loop.
- Tests: `test_joint_state.py` (4), `test_joint_scheduler.py` (4), `test_joint_loop.py` (3
  incl. THE key regression `test_sequential_factor_block_equals_standalone_run`).

## Notebooks (convention deliverables — DONE, executed with outputs baked in)

- `notebooks/execution_evolution_walkthrough.ipynb` — E0–E2 offline (15 cells):
  contract/seeds/state/BookState, freeze + poison-audit-catches-a-leak, harness +
  causality probe, jitter operator, reflection briefs, LLM prompt (canned), mini
  jitter-only run, resume, rescore.
- `notebooks/execution_evolution_live_run.ipynb` — real LLM calls on the synthetic
  panel (8 cells): brief → prompt → raw response → validated child → evaluation,
  then a 1-generation LLM-semantic loop.
- `notebooks/joint_evolution_walkthrough.ipynb` — J0–J4 offline (9 cells): ledger
  billing, 3-block round-robin run with blocks.jsonl + J trajectory, resume,
  scheduler comparison + bandit convergence plot, coupling demo, 2-fold joint
  walk-forward.

## Decisions taken during the build

- 2026-07-12: J0+J1 merged (a sequential run needs the factor→exec freeze boundary,
  so splitting would have produced throwaway code).
- 2026-07-12: full byte-identity of joint-vs-standalone factor state is impossible
  because `genome_id` embeds `uuid4` — the regression compares the invariants
  (n_trials, generation, archive factor_ids+objectives, dedup fingerprints) instead.
- 2026-07-12: refreeze book view = the controller's **accepted archive**
  (`archive_programs()`); curation-mode book views (greedy/elastic_net at each
  boundary) are a documented extension, not wired yet.
- 2026-07-12: a failed poison audit at re-freeze **raises `FrozenSignalAuditError`**
  (hard abort) — never search against leaky evaluation signals.
- 2026-07-12: joint-objective boundary scores bill `ledger.bill_look` (a VAL look
  belonging to neither hypothesis family).
