# Evolutionary Factor Researcher — Implementation Progress

Companion to `DESIGN.md`. Tracks what is **built**, what is **in progress**, and
what is **not started**, phase by phase (the phasing in `DESIGN.md §Phasing`).
Keep this in sync as the build proceeds — one line per module with its status and
test file.

Legend: ✅ done & tested · 🚧 in progress · ⬜ not started

**Post-P5 — residual-IC + regime axes and two-stage curation (Lever 2) — ✅ DONE.**
The CORE objective vector is now **5 axes**. The independence axis is the
candidate's **residual (orthogonalised) IC** — its predictive edge in the
direction the book does *not* span (novel content that actually predicts, instead
of the saturating Δ-participation-ratio, which stays available via
`EvalParams.independence_metric` / `--independence-metric delta_participation`). A
new **`regime_independence`** axis rewards *crash-complementarity*: the marginal
ΔIC on the stress bars (`--regime-kind {drawdown,volatility}`, `--regime-quantile`,
leak-safe — stress is labelled over IS∪VAL only), so a factor strong exactly where
the rest of the book is weak is non-dominated and survives selection even with
worse values elsewhere. The LOCO **marginal-value combiner now defaults to a
**nonlinear** model (`gradient_boosting`; `EvalParams.marginal_model` /
`--marginal-model {ridge,gradient_boosting,random_forest,…}`) — a *conditioning /
state* factor (the canonical low-standalone-IC volatility factor, valuable only via
`vol × momentum`-style interactions) then scores a **positive** marginal value; a
linear ridge sees only additive value and scores it ~0 (as does residual IC — pure
conditioning value is interaction value, which needs the nonlinear combiner).
Separately, **two-stage curation** decouples *what survives
selection* from *what is kept*: the controller accumulates a `kept_pool` of **every
gate-passing factor**, and `--curation {archive,greedy,elastic_net}` (default
`archive` = the one-stage Pareto behaviour) curates that pool **once at the end**
(`--n-keep N` optional) — greedy forward-selection on combined VAL IC, or
elastic-net stability selection (`research_eval/curation.py`; MCP `curate_book`) —
so a good factor is no longer discarded merely for being Pareto-dominated. Tests:
`tests/test_research_eval_curation.py`, extended `test_research_eval_{fitness,harness}.py`,
`test_evolution_{controller,loop}.py`.

**Post-P5 (round 2) — QD grid, publish-time deflation, economic reward, experience
memory — ✅ DONE (2026-07-05).** Five additions from a competitive-landscape review
(`research_docs/competitive-landscape-2026.md`; every borrowed idea is logged in
`research_docs/SOURCES.md` — keep it current). **WS1 — deflation → shared publish filter.**
The N_trials deflated-IC *gate* is removed from `harness.evaluate_candidate`/`evaluate_set`
(`deflation_ok` now always `None`; `deflated_ic` stays a diagnostic). A new
`research_eval/publish.py::publish_filter` (MCP `publish_book`) is run once at
`persist_archive` over whichever source produced the book (archive / kept-pool /
greedy-elastic / QD elites), deflating the **combined-book / marginal (LOCO)** statistic —
never standalone `|val_ic|` — and pruning by marginal contribution; `--selection-deflation
{off=discovery,on=validation}`. Search vs publish eligibility is explicit. **WS2 — QD
behavior grid.** `evolution/qd.py::QDArchive` (`--selection {nsga2,qd}`, default nsga2):
cells keyed by leak-free harness-computed descriptors (`trend_reversal`, `signal_speed`,
`stress_activation` at `--grid-dims 3`) on `FitnessResult.behavior` (NOT an objective axis —
the 5-axis Pareto is byte-identical); capped mini-Pareto per cell via the existing
crowding/`constrained_dominates`; parent sampling cell→elite with an optional AlphaPROBE
`(1−γ)^depth·(1−ω)^reuse` bias (`--depth-gamma --reuse-omega`); fixed/frozen bin edges;
`accepted_book()`/`archive_programs()` = union of cell elites. **WS3 — economic reward, no
new axis.** `cost_ok` turnover gate via `harness._turnover_netcost` (explicit
`backtesting/positions.py` construction, not `vector_backtest`) + a perturbation-fidelity
probe inside `_robustness`; `--gate-turnover --cost-rate --perturbation-weight
--perturbation-sigma`, all default OFF. **WS4 — factor-zoo dedup DIAG.** `harness._zoo_dedup`
→ max-|corr| + code distance + nearest id vs a `--reference-book`, diagnostic only.
**WS5 — per-config experience memory.** `knowledge/experience.py` (`--memory`): survivor
performance + per-mechanism attempt/survival tallies (negative evidence → exhaustion
detectable); steers seeding away from exhausted mechanisms + splices a summary into the
reflection teacher. Threaded through harness/controller/loop/CLI + MCP seam. Tests (all
green): `tests/test_research_eval_publish.py`, `test_evolution_qd.py`,
`test_knowledge_memory.py`, extended `test_research_eval_harness.py`,
`test_evolution_loop.py`. Notebooks: `…_walkthrough.ipynb` §10, `…_live_run.ipynb` §10.5.
**Still to run (billable, needs an API key):** the CLI end-to-end smoke on real S&P-100
(discovery vs validation modes) and the two-run memory experiment — the whole loop is
verified offline via the test suite, but a live LLM run is the author's to trigger.

**Learning the system:** `notebooks/evolutionary_factor_researcher_walkthrough.ipynb`
runs every component below in isolation on synthetic data (offline, no API key) —
splits/CPCV/deflation, the Pareto fitness harness, genome + mutation operators, the
reflection brief, the NSGA-II controller, RAG retrieval, the hybrid GraphRAG graph,
and a fully-offline "mini loop" that turns the real controller + harness + jitter
operator end-to-end. Read it alongside this tracker to see what each phase *does*.

**Seeing it run for real:** `notebooks/evolutionary_factor_researcher_live_run.ipynb`
is the live counterpart — a tiny end-to-end run on real S&P-100 daily bars (served
offline from the yfinance parquet cache) with the **LLM calls switched on** and the
**ML fitting real**: brainstorm → codegen → the `research_eval` reward channel
(Ridge combines the book into one forecast; LOCO marginal OOS-IC + gates) →
deterministic reflection brief → LLM mutation, then the whole `EvolutionLoop`
turning for two generations. Shows the actual intermediate I/O at every stage
(idea JSON, generated factor code, the objective vector + gate verdict, the mutation
brief, the parent→child jump, the Pareto archive). Needs `OPENAI_API_KEY`; makes a
few dozen cheap `gpt-4o-mini` calls (~1 min).

---

## P0 — Foundation (overfitting-control seam) — ✅ DONE

The riskiest-to-get-wrong layer, built first. Deterministic, no LLM. Lives in
`quant_fund_agent/research_eval/`.

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `research_eval/__init__.py` | ✅ | Package surface / re-exports | — |
| `research_eval/splits.py` | ✅ | IS/VAL/TEST `three_way_split`; CPCV `cpcv_folds` (purge + embargo); `walk_forward_folds` (anchored/rolling); `n_cpcv_paths` | `tests/test_research_eval_splits.py` |
| `research_eval/deflation.py` | ✅ | `deflated_ic` + `ic_haircut` (N_trials haircut); `expected_max_sharpe` + `deflated_sharpe_ratio` (Bailey–López de Prado DSR); norm helpers | `tests/test_research_eval_deflation.py` |
| `research_eval/fitness.py` | ✅ | `ObjectiveVector` (4 CORE axes), `GateResults`, `FitnessResult`; `dominates` + `non_dominated_front` (gate-aware); `participation_ratio`, `complexity` (AST) | `tests/test_research_eval_fitness.py` |
| `research_eval/harness.py` | ✅ | `evaluate_candidate` — signal-based deterministic fitness: Families 1–5 (standalone IC + decay, LOCO marginal ΔOOS-IC, residual IC, Δ participation ratio + max-corr penalty, CPCV robustness, coverage/degradation/deflation gates, sign consistency, parsimony). `EvalParams` knobs. Reuses `comparison/`, `backtesting/`, `modeling/`. | `tests/test_research_eval_harness.py` |

**Design-faithful choices made in P0**
- The harness is **signal-oriented** (takes DataFrames, not factor ids / LLM output)
  so it is fully testable and can never be influenced by an LLM (the core principle).
- Split seam mirrors the deployed `cutoff_date` seam: fit on **IS**, score fitness on
  **VAL** (burned by the search), CPCV over **IS∪VAL**, **TEST** untouched here.
- Marginal value (primary axis) = LOCO ΔOOS-IC of a combined model (default `ridge`)
  with vs without the candidate, measured on VAL. Empty book → the candidate is the
  whole edge.
- Deflation is **N_trials-aware** (the search count), generalising the
  `comparison/analytics` haircut from "zoo size" to "trials run".

**Deferred within P0 (documented, not yet wired)** — pick up when the owning piece lands:
- Parameter-sensitivity **plateau penalty** in robustness (needs the factor's parsed
  integer windows → arrives with the jitter mutation operator in P1).
- **Transaction-cost gate** (`cost_ok`) — left `None`; wire in P5 with the costed backtest.
- Robustness currently uses **CPCV** only; the **walk-forward wrapper** exists in
  `splits.py` but is only invoked for the thesis results pass (P5).

---

## P1 — Minimal evolutionary loop — ✅ DONE

Goal: prove evolution beats one-shot before any RAG.

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `agents/factor_research/evolution/genome.py` | ✅ | `FactorProgram` (code + hypothesis metadata incl. `expected_sign`) + `Genome` (SINGLE holds 1 program; SET-ready via `programs` list); id-invariant `code_fingerprint` dedup key; JSON round-trip | `tests/test_evolution_controller.py` |
| `agents/factor_research/evolution/controller.py` | ✅ | **Constrained NSGA-II** (Deb: feasibility → #failed gates → Pareto dominance), non-dom sort + crowding distance, binary-tournament parents, per-island populations + ring elite migration, gate-passing **Pareto archive = the accepted book**, `N_trials` counter, lineage log, save/load | `tests/test_evolution_controller.py` |
| `agents/factor_research/evolution/mutation.py` | ✅ | AST window-jitter (`jitter_windows`/`jitter_variants` → plateau probe; `random_jitter_child` → free mutation op), `rewrite_factor_id`, LLM-semantic mutation + crossover prompts (parent code + reflection brief → strict-JSON child with `expected_sign`), `parse_child_response` | `tests/test_evolution_mutation.py` |
| `agents/factor_research/evolution/reflection.py` | ✅ | **Deterministic** diagnostics→NL mutation brief (teacher channel): axes summary + rule-based advice (redundancy, conditioning-variable, OOS collapse, knife-edge windows, sign contradiction, horizon re-anchor, coverage) — no LLM writes it, identical diagnostics ⇒ identical brief | `tests/test_evolution_mutation.py` |
| `agents/factor_research/evolution/loop.py` | ✅ | `EvolutionLoop`: seed (existing brainstorm/codegen path, in-memory compile) → per-generation select/mutate/evaluate/insert; operator mix by seeded RNG; dedup skips (not billed); eval-failures not billed; per-generation checkpoints (`state.json`, `lineage.jsonl`, `run_config.json`); `persist_archive` materialises final archive via the oneshot persist path (+ evolution provenance in `metadata.evolution`) | `tests/test_evolution_loop.py` |
| `run_factor_evolution.py` | ✅ | Entrypoint; each run is a prerun under `data/workspaces/<config>/preruns/<id>/` with evolution state under `<scope>/evolution/`; oneshot baseline untouched (`run_factor_research.py`) | `--help` smoke |
| `factors/inmem.py` | ✅ | In-memory compile (full static validation, exec, registry restored) + signal computation — candidates never touch the shared package/registry until they survive | `tests/test_evolution_mutation.py` |
| `mcp/research_{service,server,client}.evaluate_fitness` | ✅ | Server-side deterministic fitness seam: in-memory signals (cached across the archive), harness call, JSON-safe `FitnessResult`; in-process fallback identical | `tests/test_evolution_loop.py` |
| `research_eval/harness.py` plateau penalty | ✅ | `evaluate_candidate(..., jitter_signals=...)`: sign-aligned VAL-IC drop under ±10% window jitter docks robustness (`EvalParams.plateau_weight`) | `tests/test_evolution_mutation.py` |

**Design-faithful choices / deviations documented in P1**
- Candidates are compiled **in-memory** (validated identically to the persist
  path) instead of being materialised per candidate — avoids file/registry churn
  for transient programs; only final-archive survivors become real files, via the
  same `materialise` + `persist_results` path as oneshot, so downstream tooling
  can't tell the engines apart.
- Reflection is a **deterministic renderer**, not an LLM call — "numbers → NL
  brief" needs no generation step, is reproducible/cacheable, and cannot flatter.
- `N_trials` is billed per **scored** candidate; children that fail to compile or
  error before producing a signal never looked at the data and are not billed.
  Duplicate programs (id-masked code fingerprint) are skipped un-billed.
- The loop is a plain Python driver rather than a LangGraph graph: the
  generation flow is a deterministic loop with no LLM-routed branching (that
  arrives with Debate in P3, which is where LangGraph topology pays off).
- Constrained NSGA-II uses Deb's feasibility rule so a population where nobody
  passes gates yet still ranks sensibly (fewest failed gates first).

## P2 — RAG — ✅ DONE

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `knowledge/embed_store.py` | ✅ | `EmbedStore`: whole-paper + chunk embedding matrices (in-memory cosine; ~1k papers needs no vector DB), disk cache keyed by corpus fingerprint + embedder name, **cutoff-date gating** (undated papers excluded under a cutoff — look-ahead safety beats recall), `verify_citations`; embedders: `openai` (`text-embedding-3-small`) / `hash` (deterministic offline fallback, `QF_EMBEDDER`) | `tests/test_knowledge_embed.py` |
| `knowledge/retrieval.py` | ✅ | `build_query` (field scope + focus + gap steering) and `retrieve_and_brainstorm` with **cardinality modes**: `1toN` (one call per paper, citations attributed authoritatively), `Nto1` (one synthesis call, ideas must fuse ≥2 papers; unverifiable grounding → idea dropped), `NtoM` | `tests/test_knowledge_embed.py` |
| loop / entrypoint wiring | ✅ | `EvolutionRunConfig.retrieval {none,rag}` + `retrieval_cardinality` + `rag_k`; `run_factor_evolution.py --retrieval --retrieval-cardinality --rag-k`; RAG grounds the seed brainstorm (per-generation Prompter retrieval arrives with P3) | `tests/test_evolution_loop.py` |

## P3 — Agent split + debate — ✅ DONE

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `agents/factor_research/debate.py` | ✅ | `run_debate`: skeptic (economic soundness / look-ahead / crowding / redundancy-vs-book, fed the data scope + book summary) → moderator (`accept\|revise\|reject`) → ≤1 revision round (unresolved `revise` → reject); **fails open** on LLM errors (quality filter, not a SPOF); full transcript for the audit trail | `tests/test_evolution_debate.py` |
| `evolution/mutation.py` hypothesis split | ✅ | `build_hypothesis_prompt` / `parse_hypothesis_response`: the LLM-semantic operator splits into Hypothesis (mechanism + `expected_sign` + `regime_dependence`, **no code**) → Debate → Codegen when `--debate on`; single-call P1 operator kept as the `off` ablation arm | `tests/test_evolution_debate.py` |
| loop / entrypoint wiring | ✅ | `EvolutionRunConfig.debate`; debate gates **seed ideas pre-codegen**, hypothesis children pre-codegen, crossover children post-hoc (accept/reject only); rejected ideas never cost codegen tokens or an `N_trials` bill; per-role models via `{HYPOTHESIS,DEBATE,CODEGEN}_LLM_MODEL[_PROVIDER]` env + `--hypothesis-model/--debate-model/--codegen-model` flags | `tests/test_evolution_debate.py` |

Note: the Prompter role (retrieval + query construction) lives in
`knowledge/retrieval.py` (P2) and the Reflection role is the deterministic
`evolution/reflection.py` (P1) — P3 adds the Hypothesis/Debate split and the
per-role model seam, completing the design's agent pipeline.

## P4 — GraphRAG (custom hybrid) — ✅ DONE

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `knowledge/graph_store.py` | ✅ | `KnowledgeGraph` (NetworkX DiGraph): namespaced typed nodes (`paper:`/`mechanism:`/`factor:`/`field:`/`anomaly:`), semantic relations (`proposes`, `realized_by`, `related_to`, `documents`, `needs`) + empirical (`corr`, `uses`); slug canonicalisation (case/plural merge); community detection (**NetworkX Louvain**, `leidenalg` used when installed — documented deviation from Leiden, agrees at this scale); JSON round-trip | `tests/test_knowledge_graph.py` |
| `knowledge/graph_build.py` | ✅ | Per-paper LLM extraction (mechanisms + required fields gated to the run vocab + anomalies + intra-paper relations), **idempotent/resumable** (ingested papers skipped, single failures logged not fatal), community summaries (deterministic default, LLM optional) | `tests/test_knowledge_graph.py` |
| `knowledge/empirical_edges.py` | ✅ | `refresh_factor_correlations` (per-underlying z-scored signal corr, thresholded, replace-not-accrete), `refresh_field_usage`, `link_factor_to_mechanism`, `refresh_mechanism_coverage` (`n_factors`/`mean_abs_ic` node attrs) | `tests/test_knowledge_graph.py` |
| `knowledge/graph_query.py` | ✅ | Global: `mechanism_gaps` (papers-but-no-factor), `computable_unexploited` (gap ∧ needed fields in scope), `under_covered_communities` (factors-per-paper thinnest first), `island_focus` (per-island steering strings).  Local: `local_context` (mechanism → papers/factors/related/fields — the N→1 bundle + provenance path) | `tests/test_knowledge_graph.py` |
| `scripts/build_knowledge_graph.py` | ✅ | Build/extend the graph over the paper library; prints the mechanism list for the human spot-check the design mandates | — |
| loop wiring | ✅ | `--retrieval graphrag`: gap-steered retrieval query + `mechanism`-tagged ideas; surviving factors linked back (`realized_by`) + field usage refreshed + graph saved — Paper → Mechanism → Factor → Field provenance closes.  Missing graph → warn + flat-RAG fallback.  `networkx>=3.0` added to requirements | `tests/test_knowledge_graph.py` |

## P5 — SET mode + thesis validation — ✅ DONE (code); ablation *runs* pending

| Module | Status | What it provides | Tests |
| --- | --- | --- | --- |
| `research_eval/harness.py::evaluate_set` | ✅ | SET fitness per the design table: axis 1 = the set's **own** combined-model OOS IC (no LOCO), axis 2 = internal participation ratio ÷ size, axis 3 = CPCV robustness of the combined signal, axis 4 = −(total complexity / size); same gates on the combined signal; reuses every SINGLE-path helper so the modes can't drift | `tests/test_evolution_set_walkforward.py` |
| `mcp/research_{service,server,client}.evaluate_set_fitness` | ✅ | SET evaluation seam (failed members dropped with a log; empty set = eval failure) | `tests/test_evolution_set_walkforward.py` |
| loop SET mode | ✅ | `--evolution-unit set --set-size N`: gen-0 seed pool partitioned into sets; **structural operators** (add/drop/replace member from the run-wide program pool), **splice** (union-sample two parents) and **member_jitter** — all programmatic/deterministic (LLM creativity enters via gen-0 members; member-level LLM mutation is a documented extension); `persist_archive` dedups shared members | `tests/test_evolution_set_walkforward.py` |
| `research_eval/deflation.py::pbo_cscv` | ✅ | Probability of Backtest Overfitting (Bailey et al. 2016 CSCV): balanced block splits, IS-winner's OOS relative rank, logit distribution + PBO | `tests/test_evolution_set_walkforward.py` |
| `evolution/walkforward.py` + `mcp/*.score_book_oos` | ✅ | The thesis results pass: per boundary date, a **fresh loop** evolves with `cutoff_date=d_i` and its archive is **touch-once** scored on `[d_i, d_{i+1})` (combined-model OOS IC fit strictly before the window + per-factor OOS ICs + PBO); `run_factor_evolution.py --walk-forward d0,d1,…`; validation only — nothing persisted to the prerun DB | `tests/test_evolution_set_walkforward.py` |

**Still to run (not code): the thesis ablation matrix.**  All arms are wired as
flags — `oneshot` (`run_factor_research.py`) vs `run_factor_evolution.py`
(`--retrieval {none,rag,graphrag}`, `--debate {on,off}`, `--evolution-unit
{single,set}`) — each arm a prerun scored by the existing comparison harness
plus the `--walk-forward` pass.  Budget/schedule those runs separately.

---

## Cross-cutting notes (added during the P1–P5 build)

- **In-memory candidate compilation** (`factors/inmem.py`) is the load-bearing
  deviation from a naive design reading: transient candidates never write files
  or registry entries; persisted survivors go through the byte-identical
  oneshot `materialise` path.
- **The evolution loop is plain Python, not LangGraph** — the generation flow
  has no LLM-routed branching (debate verdicts are handled inline); recorded as
  a deliberate simplification.
- **Trial billing rule**: `N_trials` counts *scored* candidates only; compile
  failures and dedup-skips are free (they never looked at the data).
- **Louvain instead of Leiden** for graph communities (NetworkX built-in;
  `leidenalg` auto-used when installed).

---

## Notes / running log

- **2026-07-01** — P0 built and green (34 tests). All additive; no changes to
  existing modules. Pre-existing unrelated failures in `test_data_layer.py` and
  `test_mcp_modeling.py` (a researcher factor needs a `trade` field the yfinance
  panel lacks) are untouched by this work.
- **2026-07-02** — P1–P5 built and green in one session (~70 new tests across
  `test_evolution_{controller,mutation,loop,debate,set_walkforward}.py`,
  `test_knowledge_{embed,graph}.py`; full suite green minus the two pre-existing
  failures above).  New entrypoint `run_factor_evolution.py`; new packages
  `agents/factor_research/evolution/`, `knowledge/`; `factors/inmem.py`;
  MCP research seam gained `evaluate_fitness` / `evaluate_set_fitness` /
  `score_book_oos`; harness gained the plateau penalty + `evaluate_set`;
  deflation gained `pbo_cscv`.  Changes to existing files were additive:
  `state.py` (+`FactorIdea.expected_sign`), `fitness.py` (+`from_dict`s),
  prompts/oneshot graph untouched (baseline arm preserved).  `networkx>=3.0`
  added to requirements.  Remaining work is *runs*, not code: build the
  knowledge graph (`scripts/build_knowledge_graph.py`), then the ablation
  matrix + walk-forward passes for the results chapter.
