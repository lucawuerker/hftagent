# Evolutionary Factor Researcher — Design & Architecture

Updated: 2026-07-02
Status: **P0–P5 built** — the full system is implemented and tested: the
deterministic overfitting-control foundation (`research_eval/`), the
evolutionary loop (`agents/factor_research/evolution/` + `run_factor_evolution.py`),
RAG + the custom hybrid GraphRAG (`knowledge/`), the Hypothesis/Debate agent
split, SET mode, the walk-forward validation driver and the PBO metric.  What
remains is *running* the thesis experiment matrix (each arm a prerun + the
walk-forward pass).  This document is the anchor for the work; see
`IMPLEMENTATION_PROGRESS.md` for the live phase-by-phase status and the
documented implementation deviations (in-memory candidate compilation, plain-
Python loop driver, deterministic Reflection, Louvain communities).

## Purpose

The current Factor Researcher (`agents/factor_research/graph.py`) is, in honest
terms, a **one-shot "paraphrase-a-paper-into-a-formula" generator**: it samples
papers naively, makes one LLM call per paper, generates DSL code once, records an
IC it never uses again, and keeps everything. There is no feedback, no memory, no
selection pressure, and no research-time overfitting control.

This document designs its replacement: a **closed-loop, retrieval-grounded,
overfitting-controlled evolutionary factor researcher** in the spirit of
FunSearch / AlphaEvolve, where the LLM is the *mutation operator* and a
deterministic harness is the *fitness function*. The thesis contribution is the
**agent methodology**, not realized fund P&L, so every design choice favours a
defensible experimental narrative over raw performance.

The work focuses on three levers agreed with the author:

- **A — Closed-loop evolutionary search** (FunSearch/AlphaEvolve backbone).
- **B — Retrieval (RAG) + a custom hybrid GraphRAG** for idea sourcing.
- **F — Research-time overfitting / multiple-testing discipline**, woven into the
  fitness function rather than bolted on downstream.

## The single most important principle

> **The LLM ideates and mutates; a deterministic harness scores. The LLM must
> never be able to influence its own reward.**

If an LLM-judge produces the fitness, the loop learns to write persuasive
*descriptions*, not better factors. We therefore split feedback into two
channels that must stay architecturally separate:

| | **Selection fitness** (the *reward*) | **Diagnostic feedback** (the *teacher*) |
| --- | --- | --- |
| Consumer | the deterministic evolution controller | the LLM, when mutating a parent |
| Form | numbers / Pareto vector + gate booleans | rich natural language + numbers |
| Must be | deterministic, un-gameable, out-of-sample | verbose, specific, even speculative |
| Example | "OOS marginal ΔIC = 0.012, max\|corr\| = 0.4, passes gates" | "your edge lived only in 2020 H1; 0.8-correlated with `alpha_022`; windows are knife-edge — IC halves if 20→22" |

The author's volatility example is the canonical motivation: a standalone
volatility factor has low IC, but can add real edge *in combination* (as a
conditioning/state variable inside an ML ensemble). A reward built on standalone
IC would wrongly discard it. The reward must therefore measure **marginal /
conditional value**, not standalone power.

## Locked design decisions

These four forks were decided with the author and are binding for the build:

1. **Fitness model — Pareto + hard robustness gate.** Maintain a Pareto front
   over `{marginal value, independence, robustness, parsimony}`; candidates must
   first pass hard OOS-robustness / coverage / deflation gates. No arbitrary
   scalar weights (weights can themselves overfit and bias the search).
2. **Overfit control — blocked validation stability and walk-forward.** During
   development, split VAL into contiguous blocks and re-score the fixed IS-fitted
   marginal predictions without refitting. Run the complete evolutionary process
   walk-forward for the thesis results, so every discovery is OOS to the next period.
3. **GraphRAG — custom hybrid graph.** Own store (NetworkX) fusing an
   LLM-extracted *semantic* graph (papers → mechanisms → factors) with a
   *computed quantitative* graph (factor↔factor correlation, factor→field usage).
   Off-the-shelf libraries can't easily fuse the empirical layer, and the author
   wants to learn the internals.
4. **Evolution unit — single AND set, switchable** (`--evolution-unit
   {single,set}`). SINGLE evolves one factor program, scored by what it adds to
   the book; SET evolves a whole "alpha program" (a jointly-evaluated portfolio
   of factors), closest to AlphaEvolve.

Two further points confirmed:

- **Development universe = daily SP100 first** (cheap, fast generations);
  LOBSTER / intraday reserved for a later "richness" experiment.
- **Marginal-value reference in SINGLE mode = the Pareto archive** (the current
  non-dominated front), not the full accepted book — cheaper and cleaner, and the
  archive *is* the live "accepted ensemble".

## How it grows out of the existing pipeline

This is **not a rewrite**. Today's linear pipeline becomes one *generation* of a
loop; each existing node is generalised. The `oneshot` engine stays as the
baseline arm for ablations.

| Today (`agents/factor_research/graph.py`) | Becomes |
| --- | --- |
| `load_papers` (naive `unread_first`/`random` sampler) | **Prompter / Retriever** (RAG + GraphRAG + experience memory) |
| `brainstorm` (1 LLM call per paper, one-shot) | **Hypothesis** agent (+ optional **Debate**) |
| `generate_code` (DSL + 1 retry) | **Codegen** agent (core unchanged) |
| `backtest_factors` (IC only, not a gate) | **Eval harness** (full Pareto dashboard, reuses `comparison/`) |
| `filter_and_persist` (keep-all) | **Evolution controller** (Pareto insert, islands, N_trials) |
| — | **Reflection** (numbers → NL mutation brief) feeds the next generation |

Everything new lives behind `--engine evolution`; `--engine oneshot` preserves
today's behaviour exactly.

## Feedback signal taxonomy

Six families. Tags: **[CORE]** = part of the selection fitness (Pareto axis or
gate); **[GATE]** = hard pre-filter; **[DIAG]** = computed and shown to the LLM
(teacher channel) but *not* selected on.

### Family 1 — Standalone predictive power
- IC at the factor's own horizon — **[DIAG]** (demoted from reward;
  this is the trap the volatility example warns against). `ic_own` already exists.
- Raw-signal IC stability / IC-IR — **[DIAG]**; useful teacher-channel evidence,
  but not the main robustness reward because conditioning factors can have weak
  standalone IC.
- IC decay curve across horizons — **[DIAG]**; in free-horizon mode it can suggest
  re-anchoring, while in fixed-horizon mode it tells the LLM whether to redesign
  the mechanism for the required offset.
- Quantile-spread monotonicity — **[DIAG]**.

### Family 2 — Marginal / incremental value  *(the heart of the reward)*
- **Orthogonalised (residual) IC** — **[CORE]**. Regress the new factor on the
  existing book; measure the residual's IC. "Novel predictive content the book
  lacks." A low-standalone-IC but orthogonal vol factor scores well here.
- **Marginal contribution to the combined model (LOCO)** — **[CORE, primary
  axis]**. Fit the ML ensemble with and without the factor; report ΔOOS-IC /
  ΔOOS-Sharpe of the combined signal. The most direct measurement of "adds edge
  in combination."
- **LASSO / elastic-net stability selection** — **[CORE]**. Not the single
  coefficient (unstable) but how often L1 keeps the factor across bootstrap/CV
  folds. Exactly the author's "does it survive when combined" test.
- **SHAP / tree-gain importance, incl. interaction values** — **[DIAG→CORE]**.
  SHAP *interaction* values capture "matters *conditional on* another factor" —
  the quantitative form of "volatility as a state variable."
- Permutation importance (OOS) — **[DIAG]**, model-agnostic confirmation.

> Consequence: in SINGLE mode, marginal value is measured against the **evolving
> Pareto archive**, so the fitness is *non-stationary by design* — as the front
> grows, a factor's marginal worth shifts, pushing the search toward
> diversification. SET mode sidesteps this because the unit is self-contained.

### Family 3 — Diversification / independence
- Max \|signal correlation\| to existing factors — **[CORE, soft penalty]**
  (signal-level, not IC-level), so a slightly-correlated but high-marginal factor
  can still win.
- **Δ participation ratio** of the book (effective # independent factors) —
  **[CORE]**. Already computed in `comparison/analytics.py`; reward factors that
  *raise* it. Portfolio-level, cleaner than pairwise correlation.
- Cluster novelty (opens a new cluster vs crowds one) — **[DIAG]**.

### Family 4 — Robustness / overfitting resistance  *(this is F, inside the fitness)*
- **OOS/IS degradation ratio** (`ICIR_oos / ICIR_is`, same sign) — **[GATE]**.
  You already track OOS÷IS-Sharpe in the rolling comparison.
- **Blocked validation distribution** — **[CORE]**. Reuse the exact IS-fitted
  `book + candidate` and `book only` predictions from marginal-value scoring and
  compute their LOCO IC difference on contiguous, non-overlapping VAL blocks.
  Reward high `mean − λ·std`. No model is refitted for this axis. Raw standalone
  block IC remains a model-free diagnostic.
- **Parameter-sensitivity / plateau test** — **[CORE, underused]**. Jitter the
  factor's integer windows (±10%) and measure IC stability. Overfit factors are
  knife-edge spikes; real ones sit on plateaus. *The same jitter doubles as a
  free mutation operator (see controller).*
- **Complexity / MDL penalty** — **[CORE]**. Parse the factor AST, count
  operators + free constants, penalise. Parsimony generalises and reads as
  "economically motivated, not curve-fit."
- Regime consistency (high/low-vol, bull/bear, sub-periods) — **[DIAG→CORE]**.
- **Deflated significance by number-of-trials** — **[GATE]** (see overfit
  protocol). The loop is a multiple-testing machine; deflation lives at the
  search level.

### Family 5 — Economic realism / tradeability
- Turnover & signal autocorrelation — **[CORE]**; fast-decaying signals are
  fragile and expensive, and this is a capacity proxy.
- Transaction-cost-adjusted net IC/Sharpe — **[DIAG→CORE]**; demotes micro-noise.
- Coverage / breadth (non-NaN (date,ticker) fraction) — **[GATE]**; hard floor.
- **Hypothesis–result sign consistency** — **[CORE, novel to the LLM setting]**.
  The Hypothesis agent declares an `expected_sign`; if realized IC has the
  opposite sign, that is a red flag for data-mining or broken reasoning. A
  falsifiable mechanism gives a *free* overfit detector you only get because an
  LLM stated the hypothesis.

### Family 6 — Meta signals (teacher channel only) — all **[DIAG]**
NL critique, nearest existing factors (to differentiate from), the periods/regimes
where it failed, the IC-decay curve, SHAP interaction partners, "you've already
tried 3 variants of this." This is where experience-RAG / GraphRAG feeds the LLM.

### The CORE objective vector (Pareto axes)
1. **Marginal contribution to the combined model** (primary),
2. **Independence** (Δ participation ratio; soft max-corr penalty),
3. **Robustness** (`mean_block(fixed-prediction LOCO IC) − λ·std_block −
   plateau_penalty + sign_consistency_bonus`),
4. **Parsimony** (`−complexity`).

**Hard gates** (all must pass, else the candidate is treated as dominated):
coverage ≥ τ_cov; OOS/IS degradation ≥ τ_deg with matching sign; and an optional
cost gate. Trial-count deflation is applied to the selected combined book at
publish time, not as a per-candidate gate.

## Overfitting / multiple-testing protocol (build FIRST)

The search is itself a multiple-testing machine: generate thousands of
candidates, select the best OOS IC, and you have merely moved the data-snooping
up one level. The protocol is explicit and non-negotiable:

- **Three-way temporal split.** IS → fit the ML ensemble used for marginal-value
  scoring. VALIDATION → compute fitness; *this set is deliberately burned by the
  search.* TEST → touched **once**, by the Statistician, on the final Pareto
  survivors only.
- **Blocked validation stability.** Fit the two marginal models once on IS, then
  divide VAL into contiguous blocks and compute a distribution of fixed-prediction
  LOCO IC contributions. This tests whether the measured contribution is spread
  through the validation period without conflating it with model-refit instability.
- **Walk-forward wrapper.** Re-run the *entire* evolutionary loop period-by-period
  inside the existing walk-forward harness for the thesis results chapter — every
  discovery is OOS to the next period. Used once, not every iteration (decision 2).
- **N_trials accounting / deflation.** The controller increments a counter per
  evaluated candidate; `research_eval/deflation.py` inflates the deflated-IC /
  haircut threshold accordingly (wraps the existing comparison-harness haircut
  code). This is what keeps the search honest.

This split threads through one seam, exactly like `cutoff_date` does today, so
research and deployment cannot drift.

## Agent pipeline (one generation)

State extends `FactorResearcherState`. LangGraph topology:

```
 Prompter/Retriever ─▶ Hypothesis ─▶ [Debate/Critic]* ─▶ Codegen ─▶ Eval harness ─▶ Evolution
 (RAG + GraphRAG +     (mechanism,    (adversarial      (DSL +     (deterministic   controller
  experience memory;   expected_sign, challenge;        retry,     Pareto dashboard, (Pareto front +
  data-scope gating;   horizon,       accept/revise/    unchanged) Families 1–5)     gates; islands;
  cardinality modes)   fields)        reject)                                         picks parents)
                                            ▲                                              │
                                            └────────────── *on/off ────────────┐         │
                          Reflection (numbers → NL mutation brief) ◀─────────────┴─────────┘
```

1. **Prompter / Retriever** — builds the research query from book gaps +
   data-scope (reuse the existing field-gating so only *computable* ideas
   surface) + regime/universe + (optionally) a target island/community. Pulls RAG
   papers + GraphRAG context + nearest past factors. Emits a grounded prompt
   bundle. Supports cardinality modes `1→N`, `N→1`, `N→M`.
2. **Hypothesis** — reads the bundle, emits **structured** output:
   `{mechanism, expected_sign, horizon, fields, regime_dependence,
   source_paper_ids}`. Strong reasoning model (e.g. Claude Opus).
3. **Debate** *(on/off via `--debate`)* — `proposer → skeptic → moderator`. The
   skeptic attacks economic soundness, look-ahead, likely-already-arbitraged, and
   redundancy with the book (fed by GraphRAG / experience memory). The moderator
   returns `accept | revise (≤1 loop) | reject`. **Placed before codegen** so
   tokens are not spent evaluating weak ideas; the on/off switch is itself a clean
   ablation ("does adversarial debate raise OOS quality?").
4. **Codegen** — the existing DSL generation + one-retry validator, unchanged.
5. **Eval harness** — server-side (owns the cached panel via MCP); returns the
   Pareto objective vector + gate booleans (to the controller) and the rich
   diagnostics dict (to Reflection).
6. **Reflection** — turns the cold dashboard into a natural-language mutation
   brief for the next generation (the FunSearch "feedback into the next prompt").

Per-role model selection: `--hypothesis-model`, `--debate-model`,
`--codegen-model`, so tokens are spent where they matter.

## Evolution controller

- **Genome** (`evolution/genome.py`): `SINGLE` = one factor program + metadata;
  `SET` = a list of programs. `--evolution-unit` switches; harness and controller
  are shared, only objective-1 wiring and the mutation operators differ.

  | Pareto axis | SINGLE (factor) | SET (alpha program) |
  | --- | --- | --- |
  | 1. Marginal value (primary) | ΔOOS-IC of the combined model from adding the factor to the **Pareto archive** (LOCO) | the set's **own** combined-model OOS-IC/Sharpe directly |
  | 2. Independence | Δ participation ratio; soft max-corr penalty | internal participation ratio of the set |
  | 3. Robustness | fixed-prediction blocked-VAL LOCO `mean(ΔIC) − λ·std − plateau + sign_consistency` | blocked-VAL stability of the set's fixed combined prediction |
  | 4. Parsimony | `−(n_ops + n_constants)` from AST | `−(total_complexity / size)` |

- **Selection** — NSGA-II: non-dominated sort into Pareto fronts + crowding
  distance for within-front diversity. The **archive = current Pareto front**,
  which *is* the "accepted book" used for SINGLE marginal scoring (decision: the
  Pareto archive, not the full book).
- **Islands** — K sub-populations with periodic elite migration. Each island can
  be seeded to target a different GraphRAG community, giving structural,
  mechanism-level diversity rather than only numeric diversity.
- **Parent presentation** — à la FunSearch: show the LLM a few parents sorted by
  fitness plus the Reflection brief, and request a mutation/crossover.
- **Mutation operators** (`evolution/mutation.py`):
  - *LLM-semantic* — the main creative operator (the agent pipeline above);
  - *Programmatic window-jitter* — cheap, and doubles as the robustness probe;
  - *Crossover* — SINGLE: LLM combines two parents; SET: structural add/drop/
    replace member, or splice two sets.
- **Budget / stop** — `--generations`, `--population`, or a token budget (≈$1k
  headroom available). The controller logs `N_trials` (for deflation) and full
  lineage (who mutated from whom) for the thesis.

## RAG subsystem (`knowledge/embed_store.py`)

- **Two embedding granularities** — whole-paper vectors (selection/ranking) +
  chunk vectors (GraphRAG grounding and citation checks). The **whole paper** is
  passed to the LLM (Claude 200K context), so retrieval is for *routing*, not for
  fitting context.
- **Query construction** — from under-covered mechanisms/clusters, the data scope
  (reuse field-gating), regime/universe, and the target island.
- **Cardinality modes** — `1→N` (today), `N→1` (synthesise mechanisms across
  papers — where genuinely novel factors come from), `N→M`.
- **Date-gating** — keep the existing `cutoff_date` enforcement (no look-ahead).
  **Citation verification** — a claimed `source_paper_id` must have been in the
  retrieved set, else it is a hallucination and is dropped.
- **Scale** — 1,014 papers is tiny: an in-memory embedding matrix + cosine is
  sufficient initially; no external vector DB needed.

## GraphRAG subsystem (custom hybrid)

The novelty: **fuse an LLM-extracted semantic graph with a computed quantitative
graph.**

- **Schema** (`knowledge/graph_store.py`, NetworkX). Nodes: `Paper`, `Mechanism`,
  `Factor`, `DataField`, `Anomaly`. Edges:
  - *semantic* (LLM-extracted): `Paper —proposes→ Mechanism`,
    `Mechanism —realized_by→ Factor`, `Mechanism —related_to→ Mechanism`;
  - *empirical* (computed in `knowledge/empirical_edges.py`, refreshed each
    generation): `Factor —corr→ Factor` (correlation matrix), `Factor —uses→
    DataField`, `Mechanism —coverage→ (#factors / IC)`.
- **Build** (`knowledge/graph_build.py`): LLM extraction pass over the 1,014
  papers → entities + relations → **Leiden communities** → hierarchical community
  summaries (the "global query" capability).
- **Query API** (`knowledge/graph_query.py`):
  - *Global / gap-finding* — "mechanisms with papers but no factor yet,"
    "mechanisms reachable from available `DataField`s but unexploited,"
    "under-covered communities." Drives **novel + computable** ideas and island
    seeding.
  - *Local / grounding* — around one mechanism, pull all attached papers → natural
    `N→1` multi-paper synthesis.
- **Payoff** — every factor carries a provenance path `Paper → Mechanism → Factor
  → Field`. That is simultaneously the experience memory, the redundancy check for
  the debate skeptic, and a clean explainability narrative for the thesis.

## Module layout

```
quant_fund_agent/
  agents/factor_research/
    evolution/
      controller.py      # population, islands, NSGA-II non-dom sort, N_trials, mode switch
      genome.py          # Genome = single factor program OR factor set ("alpha program")
      mutation.py        # LLM-semantic + programmatic window-jitter + structural (set) ops
      reflection.py      # dashboard dict -> NL mutation brief
      loop.py            # the LangGraph that runs one generation; driven by controller
    debate.py            # proposer/skeptic/moderator nodes (switchable)
  research_eval/
    harness.py           # deterministic fitness: objective vector + gate booleans + diagnostics
    fitness.py           # the Pareto axes + gate definitions
    splits.py            # IS/val/test + CPCV (purge+embargo) + walk-forward wrapper
    deflation.py         # deflated-IC / haircut given N_trials (wraps comparison haircut code)
  knowledge/
    embed_store.py       # whole-paper + chunk embeddings; retrieval; date-gating; citation check
    graph_store.py       # the hybrid KG (NetworkX): schema, persistence
    graph_build.py       # LLM extraction pass + Leiden communities + community summaries
    graph_query.py       # global gap-finding + local grounding query API
    empirical_edges.py   # computed Factor<->Factor corr / Factor->Field edges (refreshed each gen)
run_factor_evolution.py  # new entrypoint (or --engine on run_factor_research.py)
```

`research_eval/` deliberately **reuses** existing, trusted code:
`comparison/analytics.py` (participation ratio, clusters, LASSO/GBM importance,
haircut), `comparison/vector_backtest.py`, `comparison/standardize.py`,
`backtesting/positions.py`. The fitness harness is mostly *orchestration* of code
that already exists.

## Configuration / mode switches (one place)

```
--engine {oneshot,evolution}
--evolution-unit {single,set}
--debate {on,off}
--retrieval {none,rag,graphrag}
--retrieval-cardinality {1toN,Nto1,NtoM}
--generations N  --population N  --islands K  --budget-tokens N
--stability-blocks N  --split <IS/VAL/TEST spec>
--hypothesis-model M  --debate-model M  --codegen-model M
```

Each evolution run is a **prerun under
`data/workspaces/<config>/preruns/<id>/`** (the existing Scope/Book seam), so runs
never collide and the existing comparison / rolling harness can score them
directly.

## Thesis experiment matrix

Ablation ladder; each arm is a prerun, all scored on the **touch-once TEST set**
plus a final **walk-forward** pass:

| Arm | Claim under test |
| --- | --- |
| `oneshot` (today) | baseline |
| `+ evolution` (no RAG/debate) | does closed-loop Pareto search beat one-shot? |
| `+ RAG` | does retrieval-grounded ideation help? |
| `+ GraphRAG` | does structured gap-finding beat flat RAG? |
| `+ debate` | does adversarial critique raise OOS quality / cut overfit? |
| `single` vs `set` | factor-level vs alpha-program evolution |

Headline metrics, reported **OOS only, never IS**: combined-model OOS IC/Sharpe,
**Probability of Backtest Overfitting (PBO)**, deflated-Sharpe (N_trials-aware),
effective # independent factors, turnover/capacity.

## Phasing

- **P0 — Foundation (build first; riskiest to get wrong). ✅ DONE.** `research_eval/
  splits.py` (IS/val/test, CPCV with purge+embargo, walk-forward wrapper),
  `deflation.py` (N_trials-aware IC haircut + Deflated Sharpe), `fitness.py` (Pareto
  objective vector + gates + dominance), and `harness.py` (`evaluate_candidate` —
  the full signal-based deterministic fitness) reusing `comparison/`. No LLM changes.
  Tests: `tests/test_research_eval_{splits,deflation,fitness,harness}.py`. Deferred
  within P0: the plateau penalty (arrives with the P1 jitter operator) and the
  cost gate (P5).
- **P1 — Minimal loop.** `controller.py` + `genome.py (SINGLE)` + LLM-semantic and
  jitter mutation, driving the existing brainstorm/codegen. Goal: prove evolution
  beats one-shot before any RAG.
- **P2 — RAG.** `embed_store.py` retrieval + cardinality modes.
- **P3 — Agent split + debate.** Prompter / Hypothesis / Debate / Reflection +
  per-role models.
- **P4 — GraphRAG.** Custom hybrid graph + island-by-community seeding.
- **P5 — SET mode**, then the **walk-forward final validation** + the full
  ablation matrix for the thesis.

## Key risks

- **Marginal-value compute cost** — two LOCO fits per candidate when the book is
  non-empty. Robustness reuses these predictions and adds no model fit.
- **Fitness gaming** — keep the reward deterministic; the LLM never sees VAL/TEST
  labels.
- **GraphRAG extraction noise** — mechanism nodes can be inconsistent; needs a
  canonicalisation/merge pass and a human spot-check early.
- **Reproducibility under non-determinism** — fixed seeds, cached retrievals,
  logged lineage + N_trials.

## References

- Bailey & López de Prado — *The Deflated Sharpe Ratio* (2014); *The Probability
  of Backtest Overfitting* (2016).
- Harvey, Liu & Zhu — *"…and the Cross-Section of Expected Returns"* (2016)
  (multiple-testing critique of the factor zoo).
- López de Prado — *Advances in Financial Machine Learning* (CPCV, purging,
  embargo, labelling).
- DeepMind — *FunSearch* (Nature 2024); *AlphaEvolve* (2025).
- Shinn et al. — *Reflexion* (2023).
- Yu et al. — *AlphaGen* (2023); *AutoAlpha*.
- Lewis et al. — *RAG* (2020); Microsoft — *GraphRAG* (2024).
```
