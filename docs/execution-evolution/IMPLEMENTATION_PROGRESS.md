# Execution-Evolution Implementation Progress (E-phases)

> **Handoff document.** Keep this file current after EVERY work session — it is
> the single source of truth for "how far the build is", written so a colleague
> (or a weaker model) can pick up mid-phase. Design anchor:
> `docs/execution-evolution/DESIGN.md` (inner loop 2); the outer layer is
> tracked in `docs/joint-evolution/IMPLEMENTATION_PROGRESS.md`.
> Conventions: run tests with `./venv/bin/python -m pytest` (NOT bare pytest —
> path resolution breaks). 4 pre-existing failures unrelated to this build
> (`test_settings_env_override`, `test_fit_and_backtest_mcp_matches_inprocess`,
> `test_greedy_keeps_strong_drops_noise_and_auto_sizes`, +1): ignore them.

**Agreed build order:** E0 → E1 → E2 → J0 → J1 → E4 → J3 → J2 → E3 → E5+J4.

| Phase | Status | Notes |
| --- | --- | --- |
| E0 — execution seam + deterministic harness | **DONE 2026-07-12** | 38 tests green (`test_execution_{base,seeds,signal_freeze}.py`, `test_research_eval_exec_harness.py`) |
| E1 — jitter-only loop + resume on both loops | **DONE 2026-07-12** | `agents/execution_research/evolution/{genome,mutation,seeds,loop}.py` + `run_execution_evolution.py`; `Genome.program_type` registry (additive) in factor `genome.py`; factor `EvolutionLoop.run(resume, n_generations)` (additive, default byte-identical). Tests: `test_exec_evolution_loop.py` (10) + `test_evolution_loop.py::test_resume_continues_generations_and_trials` |
| E2 — LLM mutation + reflection | **DONE 2026-07-12** | `reflection.py::exec_mutation_brief` (rule-based advice per failure mode); `mutation.py` E2 section (`EXEC_CONTRACT`, `build_exec_mutation_prompt`, `build_exec_crossover_prompt`, `parse_exec_child_response`); loop ops `_child_{llm_semantic,crossover,jitter}` behind `p_llm_semantic/p_crossover/p_jitter` (defaults 0/0/1 → E1 byte-identical); CLI `--p-llm --p-crossover --p-jitter`. Tests: `test_exec_evolution_mutation.py` (12). **Notebooks DONE + executed**: `notebooks/execution_evolution_walkthrough.ipynb` (15 cells, offline) + `notebooks/execution_evolution_live_run.ipynb` (8 cells, real LLM, synthetic panel). |
| E3 — debate + RAG | **DONE 2026-07-12** | `agents/execution_research/evolution/debate.py`: `run_exec_debate` (skeptic attacks cost realism / capacity / leverage disguise / redundancy / backtest artifacts; ≤1 revision; fails OPEN; revised code re-enters validation; rejects never billed) + `execution_literature_snippets` (RAG splice via `knowledge.embed_store.retrieve_chunks`, fails open on empty corpus). Wired behind `ExecEvolutionRunConfig.debate/retrieval` + CLI `--debate --retrieval`. Tests: `test_exec_evolution_debate.py` (8). |
| E4 — deployment integration (executor_id) | **DONE 2026-07-12** | `StrategySpec.executor_id` + `ArchitectState.executor_id` + `StrategyRecord.executor_id`; threaded architect graph → `mcp/client.fit_and_backtest` → modeling server/service/train → `ModelStrategy`/`DynamicStrategy` (+`from_artifact`) → `backtest_strategy` dispatch (registry executor builds the book; `position_params["executor_overrides"]` overrides seed params); statistician OOS passes it; `pipeline.run_strategy_session(executor=…)`; `run_fund.py --executor` / `QF_EXECUTOR` via `execution.base.resolve_executor`. None = legacy, byte-identical (tested). Tests: `test_executor_deployment.py` (7). NOTE: evolved executors must be REGISTERED to deploy — the exec-archive materialise path (writing survivor `.py` files into an importable package) is future work (E5+). |
| E5 — exec walk-forward + ablation | **DONE (via J4) 2026-07-12** | The touch-once OOS scorer landed as `score_joint_oos(book, executor, start, end, …)` (service+server+client) — it subsumes the planned `score_executor_oos` (pass a FIXED book to score executors alone). The per-fold re-run protocol is `joint_evolution/walkforward.py::run_joint_walk_forward` (`run_joint_evolution.py --walk-forward d0,d1,…` with `--scheduler sequential --n-factor-blocks 0`-style configs covering exec-only arms). Ablation-matrix RUNS remain (thesis work, not code). |

## E0 checklist — ALL DONE (2026-07-12)

- [x] `quant_fund_agent/execution/__init__.py`
- [x] `execution/base.py` — `BaseExecutor` (stepwise `step` + optional vectorised
      `target_weights`), `register_executor`/`EXECUTOR_REGISTRY`/`get_executor`,
      `validate_weights`, `run_executor` driver, `BookState`, `executor_spec`
- [x] `execution/state.py` — `build_state_frames` (causal: vol, adv, spread?,
      per-name drawdown, signal_age)
- [x] `execution/seeds.py` — `TopKDollarNeutral` + `ZScoreThresholdEqualWeight`,
      delegating to the legacy functions (byte-equivalence by construction +
      asserted). Param overrides via instance attr `overrides` (dict). Legacy call
      sites keep calling the primitives directly until E4 flips deployment to the
      registry.
- [x] `execution/codegen.py` — `validate_executor_code` + `compile_executor_inmem`
      (registry-restoring exec + synthetic smoke via `_make_synthetic_inputs`)
- [x] `execution/signal_freeze.py` — `freeze_eval_signals(...) -> FrozenSignalSet`.
      `load()` ALWAYS reads from parquet (never in-memory frames) so consumers
      score exactly the artifact on disk. Poison audit = poison VAL rows → IS-row
      predictions must be bit-identical (proves IS-only fit).
- [x] `research_eval/exec_harness.py` — `ExecEvalParams` + `evaluate_executor` +
      `causality_probe` (truncation-replay). Slot mapping per DESIGN table; the
      5th axis is `structural_novelty` (the factor side swapped
      regime_independence → structural_novelty in a parallel worktree session on
      2026-07-11; exec reuses `harness._structural_novelty` vs `archive_codes`).
- [x] MCP: `research_service.py::{_dev_slice, freeze_signals,
      evaluate_executor_fitness(candidate, signals_manifest, jitter, archive, …)}`
      + server + client wrappers. `score_executor_oos` DEFERRED to E5.
- [x] Tests (38 green): `tests/test_execution_base.py`, `test_execution_seeds.py`,
      `test_execution_signal_freeze.py`, `test_research_eval_exec_harness.py`
- [ ] Notebook (after E2 per convention): `notebooks/execution_evolution_walkthrough.ipynb`

## Decisions taken during the build (log every deviation from DESIGN here)

- 2026-07-12: **5th Pareto axis is now `structural_novelty`** (min normalised
  code-edit distance to the nearest archive member). A PARALLEL session changed
  the factor side (`fitness.py`/`harness.py`, uncommitted at the time) from
  `regime_independence` → `structural_novelty` mid-build; the exec harness was
  adapted to match (reuses `harness._structural_novelty` against
  `archive_codes`; `evaluate_executor_fitness` gained an `archive` arg). The
  DESIGN slot table was updated. If you see `regime_independence` anywhere in
  exec code/docs it is stale — fix it.
- 2026-07-12: `DEFAULT_NEUTRAL_TOL = 0.5` (loose): the baseline top-K book is
  only approximately dollar-neutral (top-K selection breaks exact zero-sum) and
  genome #0 must satisfy its own contract; tighten per-run via
  `ExecEvalParams.neutral_tol`.

- 2026-07-11: `score_executor_oos` deferred E0→E5 (only consumed by walk-forward).
- 2026-07-11: Weight-contract convention: NaN weights are treated as 0 by the
  harness (the seeds legitimately emit NaN before z-warmup / on all-zero rows);
  ±inf or bound breaches fail validity. Documented in `execution/base.py`.
- 2026-07-11: Causality probe runs on the FIRST frozen signal only, 2 probe
  points (~1/3, ~2/3 of the dev window) — look-ahead is program-structural,
  not signal-specific; keeps evaluation cost at +2 truncated runs.
- 2026-07-11: P&L convention = `weights.shift(1) × 1-bar forward return`
  (identical to `strategy_backtester._portfolio_returns`); the last dev bar
  drops out automatically because the dev-sliced panel's forward return is NaN
  there.
