# QuantFundAgent

Master's thesis project (Mathematics & Finance, Imperial College London).

## Goal
Build LLM agents that operate like a professional quant fund: researching
features, designing strategies, and deploying them live.

## Architecture
A LangGraph multi-agent pipeline mirroring a quant fund's org chart:

- **Factor Researcher** – reads papers, brainstorms alpha ideas, generates
  factor code, backtests each one (recording its IC for reference) and keeps
  every factor that runs — IC magnitude is not a keep/drop gate. It only ever
  invents factors the *configured data feed can serve*: its brainstorm/codegen
  data-context lists only in-scope fields and any factor reading an out-of-scope
  field is dropped at persist (see the data-scope status note below).
- **Selector** – picks factors for a hypothesis.
- **Architect** – combines factors into a strategy via a refinement loop. The
  fitted model maps (cross-sectionally normalised) factor signals → forward
  returns; its prediction is the composite signal, turned into positions by the
  configured **position-construction** regime (see the status note below).
- **Statistician** – OOS tests, deflated Sharpe, accept/reject gate.
- **Portfolio Manager** – screens, allocates capital, monitors/retires
  strategies (single PM or a committee).

State flows through shared databases (factor / paper / strategy / portfolio).
The stage sequence lives in `quant_fund_agent/pipeline.py` so the notebook,
scripts, and backtests all drive the agents identically.

## Status
A working **MVP** of the full pipeline exists. Each stage and agent is
intended to be advanced significantly in future work.

**Evolutionary factor researcher — P0–P5 BUILT (thesis runs pending).** The
closed-loop, retrieval-grounded, overfitting-controlled evolutionary Factor
Researcher designed in `docs/research-evolution/DESIGN.md` (FunSearch/AlphaEvolve
spirit: **the LLM mutates, a deterministic harness scores** — the LLM never
influences its own reward) is fully implemented. **P0** `research_eval/`:
IS/VAL/TEST `three_way_split` (fit IS, burn VAL, touch TEST once) +
`walk_forward_folds`; **N_trials-aware** `deflated_ic`, Bailey–López de Prado
`deflated_sharpe_ratio`, and **`pbo_cscv`** (CSCV Probability of Backtest
Overfitting); the CORE Pareto `ObjectiveVector` is **4 axes**
`("marginal_value", "independence", "parsimony", "structural_novelty")` =
{LOCO marginal ΔOOS-IC (primary) − window-jitter **plateau penalty** −
perturbation-fidelity probe ± sign-consistency bonus (all IC-scale, folded onto the
primary axis), residual (orthogonalised) IC independence scored on IS∪VAL,
−AST-complexity parsimony, canonical-AST `structural_novelty`} behind hard
**coverage (+ optional cost)** gates (`GateResults.GATES = ("coverage_ok",
"deflation_ok", "cost_ok")`; deflation is a *publish* filter, not a search gate).
Three axes were tried and removed: **`robustness` (PSR)** (scored on the *same
reused* VAL window every generation, so it couldn't police the generational
ratchet it existed for; its plateau/perturbation/sign parts folded onto
`marginal_value`), **`regime_independence`** (superseded by `structural_novelty`),
and **`temporal_robustness`** (the raw VAL/IS degradation ratio survives as a
reflection diagnostic only, not a selection objective and not a gate).
`evaluate_candidate` (signal-based, un-gameable) + `evaluate_set`
(SET mode: the set's own combined-model OOS IC / internal PR / structural diversity).
**P1** `agents/factor_research/evolution/`: `genome.py`
(`FactorProgram` + SINGLE/SET `Genome`, id-masked dedup fingerprint);
`controller.py` — **constrained NSGA-II** (feasibility → #failed-gates →
dominance), crowding distance, tournament parents, the **two-level
mechanism-group / deme population** (below) with within-group ring migration,
gate-passing **per-group Pareto archives whose union is the accepted book** (the
SINGLE marginal-value reference), `N_trials` billed per *scored* candidate only,
full lineage, save/load; `mutation.py` — AST window-jitter (mutation op AND plateau probe) +
LLM mutation/crossover prompts fed the parent's reflection brief;
`reflection.py` — **deterministic** diagnostics→NL mutation brief (rule-based
advice; no LLM writes it); `loop.py` — seed via the existing brainstorm/codegen
path → select/mutate/evaluate/insert with per-generation checkpoints
(`state.json`/`lineage.jsonl`) and `persist_archive` through the oneshot
materialise path (`metadata.evolution` provenance); entrypoint
`run_factor_evolution.py` (each run a prerun under the workspace seam; the
oneshot baseline `run_factor_research.py` is untouched). Candidates compile
**in-memory** (`factors/inmem.py`: full validation, zero file/registry churn);
evaluation is server-side via `mcp/research_*.evaluate_fitness` /
`evaluate_set_fitness` / `score_book_oos` (cached panel + signal cache; identical
in-process under `QF_USE_MCP=0`). **P2 RAG** `knowledge/embed_store.py`
(whole-paper + chunk embeddings, cosine retrieval, cutoff-date gating —
undated papers excluded, `verify_citations`; `openai`/`hash` embedders via
`QF_EMBEDDER`) + `retrieval.py` cardinality modes `1toN`/`Nto1` (cross-paper
synthesis; unverifiable grounding → idea dropped)/`NtoM` → `--retrieval rag`.
**P3 agent split** `--debate on`: Hypothesis (structured idea incl.
`expected_sign` + regime) → skeptic/moderator debate **before codegen** (≤1
revision, fails open, transcript kept; crossover children get post-hoc
accept/reject) → Codegen; per-role models via `{HYPOTHESIS,DEBATE,CODEGEN}_
LLM_MODEL` / `--hypothesis-model --debate-model --codegen-model`. **P4 GraphRAG**
(`knowledge/graph_store|graph_build|graph_query|empirical_edges.py`, networkx):
LLM-extracted semantic layer (Paper→Mechanism→Factor→Field, slug
canonicalisation, Louvain communities — leidenalg auto-used if installed) fused
with computed factor-corr / field-usage edges; gap queries (`mechanism_gaps`,
`computable_unexploited`, `under_covered_communities`, `island_focus`) steer
`--retrieval graphrag`; ideas come back mechanism-tagged and surviving factors
link back into the graph (provenance closes); build with
`scripts/build_knowledge_graph.py`. **P5** `--evolution-unit set --set-size N`
(structural add/drop/replace from the run-wide program pool + splice +
member-jitter; all deterministic) and the **walk-forward final validation**
(`evolution/walkforward.py`: `--walk-forward d0,d1,…` re-runs the WHOLE loop per
fold with `cutoff_date=d_i` and touch-once scores each archive on `[d_i,d_{i+1})`
via `score_book_oos` — combined OOS IC + per-factor ICs + PBO; validation only,
nothing persisted). Tests (all green): `tests/test_research_eval_*.py`,
`test_evolution_{controller,mutation,loop,debate,set_walkforward}.py`,
`test_knowledge_{embed,graph}.py`. **Remaining is runs, not code**: build the
knowledge graph, then the ablation matrix (oneshot → +evolution → +RAG →
+GraphRAG → +debate; single vs set) with the walk-forward pass for the results
chapter. Live tracker: `docs/research-evolution/IMPLEMENTATION_PROGRESS.md`.

**FINAL COMPARISON RUN — prepared 2026-07-27 (code DONE, runs pending).** The
thesis comparison (ablation ladder L0–L7 on Claude Opus 5 via Bedrock + model
sweep at the full config across GPT-5.6 Sol/Luna, Claude Sonnet 5/Fable 5, Meta
Muse Spark 1.1; 2 seeds; Nasdaq-100 PIT 2010→2024-07 with a hard 2-year forward
reserve) is fully specified in **`docs/research-evolution/FINAL_RUN_PLAN.md`**
(decision record + runbook + verification gates — READ THIS FIRST before any
run). Supporting build (all default-off/byte-identical): multi-provider LLM
layer with Bedrock + base-URL wiring and a per-role **token/cost meter** with
hard `--max-cost-usd` ceiling + optional prompt transcript
(`QF_LLM_TRANSCRIPT_PATH`) in `llm.py`; `--config`/`--n-tickers 0=full` on
`run_factor_evolution.py`; per-group **archive cap** with crowding cull +
eviction lineage events; progressive-reveal drift instrumentation
(`reveal_index` stamps, rescore diagnostics + `rescore_failed` rows,
`gen_quality.jsonl`, lineage-resume + prequential-dedup fixes) + a byte-exact
LOCO **fit cache**; fail-closed curation/publish (`QF_PERSIST_FAIL_OPEN=1`
escape); `--creative-frac` ungrounded-idea operator + relaxed (non-economic,
2–4 sentence) mechanism briefs; `--memory-key` per-arm memory isolation;
paper harvester restructured into scope-tagged query blocks
(fundamental/general/price) with `allowed_scopes` retrieval masking and a RAG
paper-text cap; orchestrator `run_ablation_matrix.py` + `matrix/final_matrix.yaml`
(resumable, preflight probes/mask-density/forward-reserve asserts); walkthrough
notebook scaffold `notebooks/final_run_walkthrough.ipynb`; run configs
`quant.config.{nasdaq100_2010,sp500_2010}.yaml` (panel verified 3,665×209,
density 0.463). **Before any credits: build the +1000-paper corpus, re-embed
with `QF_EMBEDDER=openai`, build the knowledge graph (missing → graphrag arms
crash), then the plan's D1–D5 gates.** Revisions same day: marginal combiner is
**`lightgbm`** (0.86s vs sklearn GB 121.9s per fit at run scale, same nonlinear
interactions); archive cap **40 factors/group**; curation **`archive`** (per-group
front-1 survives; greedy stays optional); **the 101 Kakushadze formulaic alphas
are now COMPLETE** (15 IndNeutralize alphas implemented with FMP sector/industry,
subindustry→industry fallback, eleven missing ×−1 signs restored vs the stale
deferred table; `factors/_labels.py` helper; allow-listed for in-memory compile)
and form the standing fixed book via `scripts/build_formulaic_prebook.py` →
`data/prebooks/formulaic_101.json` (101 members validate) wired as
`--fixed-book`/`--reference-book` in the matrix; researcher prompts sharpened to
a novelty/falsifiability/PM-grade-skeptic bar (+7% tokens). **2026-07-28
hardening (B15–B17 in the plan):** `--mechanism-groups-mode max` (the group
count is a hard UPPER limit — the run shrinks to however many usable graph
communities exist; matrix asks for 8/max); orchestrator **per-entrypoint
defaults** (`gp_defaults`/`oneshot_defaults` in the plan YAML — evolution flags
no longer leak into the GP/oneshot argv and crash them at L0/L1), GP `--config`
flag + preflight `load_panel` call fixed, `--n-tickers 0`=full now honoured by
`run_gp_factor_mining.py`/`run_factor_research.py`/`run_evolution_timing.py`
too (0 used to slice the universe EMPTY); plan-vs-argparse guard
`tests/test_final_matrix_plan.py`; re-embed CLI `scripts/rebuild_embeddings.py`
(runbook step 2 was pseudocode). Preflight verified live offline
(`--preflight-only --no-probes`: panel 3,665×209 density 0.463, forward reserve
OK, correctly refuses on the missing knowledge graph). Full test suite now
**genuinely green (679 passed)** — fixed the two pre-existing `test_data_layer`
failures (ambient `quant.config.yaml` leaking into tests that assume lobster
defaults) and an order-dependent `test_mcp_modeling` failure
(`comparison.load_panel_cached` leaks `_PANEL_CACHE`/`_SIGNAL_CACHE`/
`QF_DATA_TICKERS` etc. process-wide; `test_comparison.py` now snapshots and
restores).

**FINAL RUN — EXECUTION STARTED 2026-07-28 (Anthropic API, $500 credit).** The
ladder now runs on **Claude Opus 5 via the direct Anthropic API** (NOT Bedrock)
from `matrix/opus_ladder.yaml` — `budget_usd: 440` enforced GLOBALLY by
`run_ablation_matrix.py` (cumulative spend read from every arm's checkpointed
`llm_usage.json`; each launch's `--max-cost-usd` clamped to the remainder;
sweep stops when exhausted; full seed-0 ladder ordered before any seed-1 arm).
Prerequisites DONE: +709 papers harvested (corpus 1,723; scopes fundamental/
general; legacy `data_scope=None` always passes the retrieval mask); OpenAI
re-embed (`text-embedding-3-small`, token-aware batching + tiktoken per-input
truncation — the old char-based batching 400'd); knowledge graph built with
**claude-haiku-4-5** (~$13, 1,723 papers → 2,295 mechanisms, resolves the FULL
8 mechanism groups → `children-per-deme` trimmed 4→3 per plan §C;
`build_graph` now checkpoints every 25 papers + stops cleanly on
`LLMBudgetExceeded`). D2 timing: **9.6s/candidate** all-in on the full 209-
ticker panel (lightgbm combined fit 1.4s). D3 graphrag smoke + D4 kill/resume
verified (lineage/prequential no dupes). **Critical launch-blocking fixes made
in-session:** (1) Claude 4.7+/5-family REJECTS `temperature`/`top_p`/`top_k` —
`make_chat_llm` now strips sampling params for those models
(`_NO_SAMPLING_MODELS`); (2) Anthropic thinking returns `content` as a LIST of
blocks — `_TextContentModel` wrapper flattens to `str` for every
`resp.content` consumer; (3) debate was killing 100% of ideas: the 4o-mini-
class moderator invents look-ahead objections (PIT fields; causal ffill) and
rejects on empirical-merit grounds — the moderator prompt now states the PIT
data guarantee and reserves "reject" for structurally fatal flaws (harness
judges merit, per FunSearch division of labour), and an unresolved revise now
FAILS OPEN (accept latest revision) — verified with Opus 5 as judge: correct
causality reasoning, accepts-with-revisions (~$0.3/idea); (4)
`codegen._make_synthetic_panel` lacked ALL fundamentals fields so every
fundamentals factor died at validation with KeyError — now includes the full
~137-field archive vocabulary (labels as strings, quarterly-stepped numerics)
+ `vwap`/`returns`; (5) evolution entrypoint persists the archive on
`LLMBudgetExceeded` (never discards paid work) and exits rc=3 when a run
scored ZERO candidates (orchestrator must not mark it ok); (6) oneshot
`run_factor_research.py` gained `--max-cost-usd` + budget-graceful stop +
`llm_usage` in manifest. NOTE: this environment kills background tasks every
~30 min — the orchestrator is simply relaunched (arms auto-resume from
checkpoints; completed arms skipped).

**Non-evolutionary "refine" variant + L4R comparison arm (built 2026-07-31).**
To test whether the evolutionary machinery earns its cost with strong models,
`run_factor_evolution.py --variant refine` (default `evolve` = byte-identical)
swaps ONLY the operators: every seeded factor starts a *lineage* refined by the
LLM against its own deterministic evaluation report at most `--refine-rounds`
(default 2) times via the new `build_refine_prompt` (*same factor, same
mechanism, better implementation* — mechanism switches forbidden, unlike the
mutation prompt); a deme out of refinement work re-seeds fresh graphrag-grounded
ideas every generation (coverage keeps broadening); the only combination
operator is the occasional explicit cross-group synthesis (`--p-cross-group`,
parents via the existing per-group Pareto tournament; children start their own
lineage). No same-group crossover, no jitter, no migration, no tournament-bred
descent; harness scoring, 4-axis Pareto, gates, group archives + cap,
progressive reveal (refine briefs are refreshed from the post-rescore fitness),
N_trials billing, curation and publish deflation are identical. Resume-safe via
`evolution/refine_state.json` (single-unit only; SET mode raises). Failed
refinements still consume a round. Tests: `tests/test_evolution_refine.py`;
`tests/test_final_matrix_plan.py` now also guards `matrix/terra_l4.yaml` +
`matrix/terra_l4_refine.yaml`. **Comparison run**: `L4_terra_s0` (GPT-5.6
Terra, full L4 config) finished 2026-07-30 — 44 factors, 798 trials, $86.70,
903 min; the ablation arm `L4R_terra_s0` (`matrix/terra_l4_refine.yaml`: same
config/seed/reveal schedule, `variant: refine`, `children-per-deme 1` →
deliberately cheaper, budget $120 from the OpenAI pool) **completed 2026-08-01:
62 factors, 468 trials, $59.71, 329 min**, full post-analysis suite run.
Headline (`docs/research-evolution/L4_VS_L4R_COMPARISON.md`): at 69% of the
cost the refine arm's honest OOS record (prequential blocks, TEST tail,
forward-reserve Sharpe 0.66 vs 0.47, DSR prob 0.77 vs 0.50, cross-sectional)
matches or beats evolution, with a much smaller in-panel→forward overfitting
gap — while evolution wins every in-search metric and the per-underlying
construction. Decision record in
`docs/research-evolution/L4R_REFINE_ARM_DECISIONS.md`. Side hardening from
this run: generation-0 seeding now checkpoints per mechanism group and resumes
mid-seeding (kill-resilient; `_admit_seed_group`); the figure suite's category
palette gained `carry` + `.get` fallbacks.

**Factor-book cross-analysis Terra-L4 vs Opus-5 (2026-08-04).**
`scripts/analyze_l4_factor_books.py` + `plot_l4_factor_books.py` →
`data/comparisons/l4_factor_analysis/` (REPORT.md + 9 figures + CSVs): pure
factor-level comparison of `L4_terra_s0` (44 F.) vs `L2_opus5_s0` (18 F. — the
only completed Opus evolution run; **there is NO Opus L4**, the $600 Anthropic
budget died after L2, so model and ladder rung are confounded) vs the 101
formulaic alphas, on the forward panel with the runs' own 60/20/20 split
(DEV→2021-08, TEST→2024-07, FORWARD→2026-07). Headline: Opus book = best OOS
generalisation (combined lightgbm TEST pooled IC 0.045 vs Terra 0.024 / zoo
0.026; only book with positive cross-sectional OOS IC; eff. N 12.7/18); Terra
book = marginal-value play (alone at zoo level, but lifts zoo FORWARD 0.024→
0.041; biggest diversity add, median max|ρ| to zoo 0.28); the two books are
near-orthogonal (cross ø|ρ| 0.056). All three books' cs-IC collapses jointly
at the 2021 TEST boundary (regime shift, not pure overfit). 4/44 Terra factors
are degenerate on the forward panel. Best single factor OOS: Opus
`comparable_basket_repair_rate_activity` (TEST 0.062). Extended same day:
Terra+Opus combined (62 F.) TEST 0.041 / FWD 0.039, +101 alphas (163) 0.039/
0.040 — the union beats either book alone; the DEV→TEST combined-IC collapse
is a combining-MODEL artefact, not factor overfitting (per-factor ICs hold or
rise OOS). English editable report (no cs-IC columns, per user):
**`factor_book_analysis.tex` → `.pdf`** (compile with `tectonic`, brew-installed
2026-08-04; hand-editable LaTeX, no AI traces incl. PDF metadata) +
`figures_en/` via `scripts/plot_l4_factor_books_en.py`; a `.docx` variant
(`scripts/build_l4_report_docx.py`) also exists.

**Combiner-model study (2026-08-05).** Follow-up on the factor-book analysis:
the combined-book IS→OOS collapse is a COMBINING-MODEL property, reproduced on
the two finished server WF arms (L2WF 19 F. / L4WF 57 F., pulled locally;
per-factor mean |IC| holds 88–92 % OOS while combined LightGBM retains 8–20 %).
`scripts/analyze_combiner_models.py --panel forward|wf` +
`analyze_wf_block_metric.py` (+`plot_combiner_models.py`)
→ `data/comparisons/combiner_models/` (REPORT.md links the artifact page).
**Metric (user decision same day): the WF OOS statistic is the MEAN of the 10
per-block ICs** (one IC per walk-forward step, the prequential convention),
with the in-sample side measured the same way (fit window chunked into 126-bar
blocks) — blockwise ICs run systematically HIGHER than long-window pooled ones,
so both sides must use the same metric. Under it: **ridge is the best WF
combiner for every book** (ø block-IC 0.053–0.067, 10/10 positive blocks,
half GBM's block std; L2WF: ridge 0.053 / lasso 0.055 vs GBM 0.015 at 5/10)
and beats the runs' own prequential refit record (L4WF 0.064 vs 0.035, L2WF
0.053 vs 0.020 — partly book-maturity, the record traded the then-current
archive). λ₂≈N…10N improves GBM where it's weak (union 0.045→0.058) but never
reaches ridge on the WF panel; on the forward panel (window metric, 163-F.
pool) GBM+λ₂=N stays best (TEST 0.0448 vs ridge 0.020). Lasso caveat: usually
keeps only 1–8 factors (Opus book the exception: 16/18); RidgeCV picks α=10⁴ =
the grid edge in EVERY fit (grid should extend 10⁵–10⁶). Server book code
synced into the local researcher package (load_book remaps server code_paths
by basename).

**Knowledge-graph factor link-back + frozen thesis snapshots (2026-08-13).**
The published books of ALL completed LLM arms (L1 oneshot Opus/Terra s0+s1,
L1H/L1HB, L2/L2WF/L2WFB/L2WFP, L4/L4R/L4D/L4IC, L4WF–L7WF, GLD HF; GP and the
incomplete L4RB excluded) are post-hoc linked into `data/knowledge/graph.json`
via NEW `scripts/link_factors_into_graph.py`: 44→**859 factor nodes** (node
attrs `preruns`/`engine`), +65 `realized_by` edges (mechanism from the
program's own stamp or inherited up the genome parent chain — edge attr
`provenance: seed|inherited`; stamps are sparse in the checkpoints, only
seeds carry them), `uses` edges from `required_inputs` (fields 55→128, uses
edges 4,162), coverage stats refreshed over 768 factor ICs; gap queries +
`mechanism_group_specs` still resolve 8 groups. The script links ADDITIVELY —
in-run `refresh_field_usage` drops every existing `uses` edge and had
silently clobbered the ladder snapshot's 71 uses edges in the local graph
(repaired via `--merge-uses-from`). Updated graph deployed to lagias
(identical 5,118 nodes / 12,527 edges). **Frozen thesis snapshots** in
`data/knowledge/frozen/` (local + mirrored on lagias, see its README):
`graph_wf_ladder_snapshot_2026-08-01.json` = the exact `--graph-readonly`
snapshot every WF-ladder arm resolved groups from, and
`graph_local_pre_linkback_2026-08-13.json` = local pre-link-back state. Any
future arm that must stay comparable to the s0 ladder should point at the
frozen snapshot, NOT the live graph (factor coverage now shifts the group
ranking).

**LLM-comparison plan CREATED (2026-08-13, runs pending keys).**
`matrix/llm_comparison.yaml` — the final experiment: 10 research LLMs × 2
seeds on the L1HB shape (192 graph-grounded seeds, children-per-deme 0, full
WF scoring, graph-readonly; seed-matched, cost reported; global cap $500,
$40/arm). Providers: luna/terra/sol (OpenAI key, luna runnable NOW),
fable5/opus5/sonnet5 (Bedrock via AWS credits — keys pending),
grok-4.6 + deepseek (Azure AI Foundry OpenAI-compatible base-URL envs —
pending; deepseek native fallback), muse (META_API_KEY pending), gemini
(google_genai — pending; pin exact ids for grok/gemini/deepseek when keys
arrive). Plan guarded in `tests/test_final_matrix_plan.py` (34 passed);
`--preflight-only --no-probes` green. Target box lagias (SSH still down
2026-08-13 — check before launch). BEFORE first launch: fix
`_residual_ic` pairwise-coverage bug. Launch: `run_ablation_matrix.py --plan
matrix/llm_comparison.yaml --only LC_luna_s0`.

**L2WFP COMPLETE — retrieval cleanly priced (2026-08-12 late).** Seeded
96/96 (chunked fix verified live), 20 gens, 27 F., 856 trials, $105.66:
prequential **+0.0286** ≈ baseline +0.035 (within noise) — but the weakest
per-factor quality of the family (median |IC| 0.0045 vs 0.0114; 44% sign
flips vs 18%; eff N 14.5 vs 22.0; a max|ρ|=0.99 near-duplicate pair) and
**PIT best only 0.038 (lasso) vs baseline 0.073** — retrieval grounding
doesn't move the noisy headline record, it ~doubles the deployable combined
IC. Full analysis in `wf_arm_analysis_local/L2WFP_terra_s0/` + PIT race;
ladder (12 rungs), v2 PDF and artifact all final.

**L4IC result + L2 seeding bug/fix + L2WFP relaunch — 2026-08-12.**
**L4IC_terra_s0 COMPLETE** ($117, 849 cand., all 20 gens): per-factor =
textbook winner's curse (archive collapsed to 1 factor/group — a
single-objective front is one point; promised median |VAL IC| 0.040 → 0.025
realised; 5/8 survivors' VAL sign disagrees with their own fit window); per
book = surprising insurance (8 near-orthogonal survivors, static linear
0.082 10/10; kept-pool 846 F. PIT ic 0.070 vs baseline 0.073). Reading in
the v2 PDF: 4-axis+reveal make individual factors trustworthy; diversity
structure+linear combining protect the book. **L2WFB_terra_s0 (parity rerun)
FAILED ITS PURPOSE**: seeded 10/96 — with retrieval none the seeding made ONE
86-idea brainstorm call which timed out and was skipped fail-open
(`loop.py` `_safe_brainstorm` single-call path). Kept as a no-retrieval
replicate (preq +0.012, 14 F., $99 — confirms L2's weakness but ALSO
confounded). **FIX**: non-retrieval seeding now chunks into ≤12-idea calls
with one retry each (loop.py); **L2WFP_terra_s0** (true parity, 96 seeds)
launched via `scripts/l2wfp_chain.sh` (chained behind L2WFB book analysis).
Per-factor stats switched to ABSOLUTE IC everywhere (user: negative IC is
predictive) — under |IC| L5/L7 medians BEAT L4 (0.0146/0.0160 vs 0.0114) but
with worst sign stability (31–33% flips vs 18%) and ~25% ρ=1.0 level-like
members; L4 leads sign stability + combinability. Ladder/figures/PDF/artifact
all updated. lagias SSH STILL down 2026-08-12.

**Book-analysis v2 (baseline+ablation restructure) + L4IC harness-ablation arm
— 2026-08-10.** User feedback on the 4-arm report: restructure around ONE
baseline (L4WF) discussed in depth, then ablation stages showing where value
comes from. Done: `scripts/wf_book_analysis_figures2.py` → `figures2/` +
`wf_book_analysis_v2.tex→.pdf` (baseline deep-dive b1–b4, ablation-ladder
chart l1 with prequential ±SE + PIT-best per arm, cross-arm l2) and
`derived/ladder_summary.csv`; the ablation-QA arms (L1H/L1HB/L4D) got the full
local book analysis (`wf_arm_analysis_local/<arm>/` + PIT races; L0WF factor
code not mirrored locally → skipped, reported via its prequential record as
metric-gaming evidence). **NEW seam `--objective ic`** (`EvalParams/
EvolutionRunConfig.objective_mode`, threaded client→service→harness; test in
`test_research_eval_harness.py`): candidate scored by standalone |VAL IC|
ONLY — other axes constant 0.0 → NSGA-II degenerates to IC ranking; default
byte-identical. **Arm `L4IC_terra_s0` LAUNCHED locally** (chain
`scripts/ablation_analysis_then_l4ic.sh`, log `data/ablation_analysis_l4ic.log`,
caffeinate-detached): L4 config but `--objective ic` AND no progressive reveal
(classic IS/VAL/TEST on `quant.config.nasdaq100_2010_to2021.yaml`) so 2021-26
stays an untouched holdout for the post-hoc block comparison — quantifies what
the 4-axis + progressive-reveal harness itself contributes; max-cost $180.
NOTE 2026-08-10: lagias SSH times out (HTTP still up) — everything ran
locally; server data (L5/L7 usage, L1WF per-factor) unreachable this session.

**WF-ladder factor-book analysis — DONE 2026-08-09.** The Opus-vs-Terra-style
book analysis rerun for the four WF ladder arms L2/L4/L5/L7 (L0/L1/L6-set
excluded, user request): `scripts/wf_book_analysis_{derive,figures}.py` →
`data/comparisons/wf_book_analysis/` (raw/ pulled from lagias, derived/ CSVs,
figures/, supervisor report `wf_book_analysis.tex→.pdf` — tectonic, clean
metadata, English, figures + 1–2 sentences + key learnings; artifact page
published same day). Covers per-factor fit-vs-WF block ICs, combined-book
combiners (internal prequential LightGBM vs PIT-lasso/static ridge/lasso/GBM),
final-archive Pareto-axis trajectories from lineage rescore rows, eigenvalue
participation-ratio diversity, and PIT-lasso selection (per block + 3 phases).
Headlines: per-factor |IC| fully retained OOS in every arm (IS→WF corr
.83–.94; L5 .24); the internal LightGBM refit is the WEAKEST deployment
combiner everywhere (L4 PIT-lasso 0.072 vs prequential 0.035); L4 best on
every measure (57 F., eff. N 22, median +0.008); lasso churns hard (37–67%
never selected) with stable ICs (37/38 blocks positive). **BUG found: the
residual-IC independence axis was inactive for ~97% of candidates in L5/L7**
— `harness._residual_ic` needs ≥30 rows where the candidate AND every book
member are finite, so one sparse-coverage member (L5/L7 admitted 28–40%-
coverage factors at gen 0–1) returns None for everyone; the debate arms
effectively evolved on a 3-axis Pareto (ablation confound; fix = pairwise-
complete orthogonalisation). Archive complexity also ratchets up ~2× over 20
generations in every arm while novelty stays flat.

**LLM-contribution ablation — RUNNING 2026-08-09 (see
`docs/research-evolution/ABLATION_QA.md` — READ THAT for design, data
inventory and results).** Decomposes L4WF's edge along ideation → +det.
scoring (L1H, `children-per-deme 0` seam) → +det. evolution (L4D, jitter-only
via `--p-llm 0 --p-crossover 0 --p-cross-group 0 --p-jitter 1`) → +LLM
evolution (L4WF), plus L0WF = GP with progressive reveal **ported into
`gp/loop.py`** (GP state under `prerun/gp/`, not `evolution/`). Plan
`matrix/ablation_qa.yaml`. **First result: L1H (ideation + deterministic
scoring, NO evolution) matches L4WF's honest walk-forward record — prequential
mean IC +0.032 vs +0.035, 80% hit both — at $9.53 vs ~$250 and 81 vs ~800
trials.** Heavy compute moved to the local M2 (Hostinger steal throttle
re-engaged repeatedly; L0WF still on lagias); L1HB (seed 24/group → target
40-50-factor archive) queued behind the L1H PIT race. s1 ladder arms on HOLD.

**WF post-analysis campaign — COMPLETE 2026-08-08.** All 15 books (L1WF–L7WF
±zoo, unions, zoo) carry the full 9-method × 10-block PIT matrix with saved
weights/models/OOS-predictions + shared signal store; the driver exited
cleanly. Final leaderboard (mean WF block IC across books): lasso 0.071 >
ic 0.064 > ridge 0.063 > autoalpha 0.057 > rf 0.055 > lightgbm 0.050 ≫ kaku
0.035/0.025; wins lasso 10/15, autoalpha 3 (zoo + both L7WF books — the
memory arm is the one evolved book where the rank ensemble wins), ic 2.
**L7WF_terra_s0 finished 2026-08-07** (20 gens, 42 factors, $205.62, full
prequential record; the multi-day gen-5/6 stall was 50% hypervisor CPU-steal
from the provider fair-use throttle — twice — plus box contention, not the
arm). Strategy lab `scripts/wf_pit_strategy_lab.py` (tune 2021-23 / validate
2023-26): frozen per-name construction (rolling-252 z → EWMA hl6 → |z|>0.5
band → λ=0.15 partial adjust + no-trade band → 10% vol target) + EW-index
hedge capitalises the LINEAR combiners' per-underlying IC: L4WF/lasso 1.19–
1.39, L4WF/ic 0.99, union/ic 0.92 net Sharpe at 0.002–0.01/day turnover;
cs-MN construction instead monetises rf (union 0.92, L4WF 0.69) and inverts
the linear ranking; zoo/+zoo books need per-name beta-hedging (open). Report
artifact (claude.ai, 15-book matrix + lasso sparsity/turnover/never-selected
analysis) maintained from `scratchpad build_report.py`. s1 ladder arms on
HOLD via `after: HOLD_S1` (L1WF s1 slipped through a stale in-memory
orchestrator plan and completed; stale orchestrator killed, holds enforced).

**WF post-analysis suite — launched on the server 2026-08-05 (autonomous).**
Detailed analyses for every finished WF-ladder arm (L1WF oneshot s0, L2WF/
L4WF/L5WF/L6WF s0; L7WF + s1 arms picked up automatically when they finish),
running in tmux session `analysis` on lagias **inside `lagias-research.slice`**
(so the CPU quota re-applies once L7WF s0 restores it), driver
`scripts/run_wf_post_analysis.sh` → `data/comparisons/wf_arm_analysis/` (on the
server; rescans every 30 min, exits after L7WF s0 is analysed). Two parts:
**(1) `scripts/wf_arm_factor_analysis.py --arm <prerun|zoo>`** — per-factor
pooled per-underlying IC on the fit window and each of the 10 prequential
126-bar blocks (block mean/std/hit + sign-consistent retention vs a
trailing-126-bar in-sample block metric), book diversity (mean |ρ|,
participation-ratio effective N), static combined fits (ridge/lasso/lightgbm,
fit < 2021-07-20, scored per block) and the run's own prequential record →
`<arm>/{per_factor_blocks.csv,combined_static.csv,diversity.json,REPORT.md}`.
**(2) `scripts/wf_pit_combiner_study.py --arm <a[+b…]>`** — the point-in-time
WF combiner race (user protocol 2026-08-05): every 126-bar block from
2021-07-20 the combiner is REFIT on all prior bars using ONLY the factors that
existed then (evolution arms: `replay_snapshots` archive at gen g trades block
g+1, SET genomes contribute all members; oneshot/zoo: full book) and scored on
the block. Methods: equal / ic-weighted / RidgeCV (grid extended to 1e6) /
LassoCV / LightGBM / **`kakushadze`** (Kakushadze & Yu *How to Combine a
Billion Alphas*, arXiv:1603.05937 — NOT "million"; eq. (1) w∝C⁻¹E on daily
long-short alpha-P&L streams with an eRank-capped PC factor-model covariance,
SMW inverse — the paper's own prescription for N≪T) / **`kaku_reg`** (their
verbatim Sec. 5.3 weighted-regression recipe incl. overall-mode removal;
built for N≫T, lookback shrunk to N/2 and `regime_ok` flagged) /
**`autoalpha`** (arXiv:2002.08245: top-150 by mean daily cs-IC, sign-flip,
greedy first-PC PCA-similarity gate <0.9, LightGBM lambdarank + XGBoost
rank:pairwise [xgboost 3.3.0 installed server-side] with daily query groups,
5-bin labels, z-scored score-average ensemble; omitted hyperparams per
docstring). Books: each arm alone, each arm+zoo, the 4-arm s0 union ±zoo.
Resume-safe jsonl per label + `<label>_summary.csv`; cross-arm
`SUMMARY.md`/`ALL_PIT_SUMMARY.csv` rebuilt each pass. Sources logged in
`research_docs/SOURCES.md`. Local smoke (L2WF g11): PIT snapshot 15 factors,
kaku_reg 0.038 / kakushadze 0.020 / ic 0.017 / autoalpha 0.005.

**WALK-FORWARD TERRA LADDER — LAUNCHED 2026-08-01 on the Hostinger VPS
(`ssh lagias`, 8 vCPU/31 GB, repo at `/root/QuantFundAgent`).** The ablation
ladder now runs on **GPT-5.6 Terra** (OpenAI pool, global cap **$2,000**) from
**`matrix/terra_wf_ladder.yaml`** with a NEW two-phase progressive-reveal
schedule (user decisions 2026-08-01): panel extended to **2010→2026-07-27**
(`quant.config.nasdaq100_2010_wf.yaml`, 4,165 bars × 232 tickers, density
0.420), **NO forward reserve and NO test tail** — the OOS evidence is the
schedule itself. `--wf-blocks 10 --wf-block-bars 126`
(`progressive.build_schedule` two-phase branch + `_build_two_phase_schedule`):
generations 1–10 reveal 2015-03-16→2021-07-20 in ~160-bar blocks (reveal-every
1; seed window 2010→2015-03-16 = 45% of PHASE-1 span), generations 11–20 each
reveal one 126-bar (~6-month) block of 2021-07-20→2026-07-27 that is
**prequentially scored (traded) BEFORE the archive may adapt to it**
(`_advance_reveal` order was already prequential→advance→rescore) — a 5-year
live walk-forward record per arm in `prequential.jsonl`. Incompatible with
`--final-holdout` (raises). Supporting changes: **`--graph-readonly`**
(`loop.link_programs_into_graph(readonly=)` — no factor-provenance write-back,
so every arm ranks mechanism groups from the identical graph snapshot; the
ranking is factor-coverage-based and 36 factor nodes from earlier runs already
sit in the graph), plan key **`allow_no_forward_reserve`** (explicit preflight
opt-out), **per-arm `config:` override** in `run_ablation_matrix.py`, and
`_reveal_index` now reads the schedule (source of truth) instead of the cadence
formula. Arms (all Terra-L4 setup: 8 groups max × 3 demes × 2 children, 12
seed-ideas/group, archive-cap 40; full s0 ladder before s1): L1WF oneshot on
**`quant.config.nasdaq100_2010_to2021.yaml`** (2010→2021-07-19 = exactly the
evolution arms' pre-WF data; its book is to be forward-deployed 2021-07-20→
2026-07-27 post-hoc with periodic deterministic re-weighting/re-curation — no
new factors), L2WF (retrieval none, 1×4×12), L4WF, L5WF (debate), L6WF (set 3),
L7WF (memory). **No L0 (GP) and no L3 (rag)** — both deliberately dropped
(user 2026-08-01; GP deferred, graphrag is the only retrieval arm kept). The
old L4RB_terra_s0 refine-broad run was stopped locally mid-run (user will
finish it later; checkpoint intact). Server: tmux sessions `ladder`
(supervisor while-loop relaunching the orchestrator every 300s —
`data/wf_ladder_orchestrator.log`), `status` (regenerates
`scripts/matrix_status.py` HTML every 60s), `httpd` (port 8899, ufw-opened);
phone dashboard at `http://31.97.141.166:8899/wf-2b505d86a0f2/`. Tests:
5 new two-phase cases in `tests/test_evolution_progressive.py`, graph-readonly
cases in `test_evolution_mechanism_groups.py`, plan guarded in
`test_final_matrix_plan.py`. **Multi-lane parallelism (same day):**
`run_ablation_matrix.py` gained an atomic per-arm `orchestrator.lock`
(O_EXCL + stale-pid reclaim, released in `finally`) and an `after:` arm key
(dependency not `ok` → `[blocked]`, retried on the next supervisor pass), so
N supervisor lanes can run the SAME plan concurrently and self-partition the
arms — the server runs tmux `lane-a` + `lane-b` via
`scripts/ladder_lane.sh <lane> [delay]` (two arms in flight at all times,
separate logs `data/wf_ladder_lane{A,B}.log`). `L7WF_terra_s1` declares
`after: L7WF_terra_s0` (shared `memory-key` — the cross-run memory ablation
needs s0's experience persisted first and must not race). Caveat: with
concurrent lanes the global budget clamp is per-launch, so total spend can
overshoot `budget_usd` by at most the in-flight arms' `max-cost-usd` (bounded
at $250/arm; irrelevant at expected ~$700–900 total spend). Lock/blocked
tests appended to `tests/test_final_matrix_plan.py`.
**Rescore-cost hardening (2026-08-03).** By generation ~8 the every-generation
archive rescore had blown up to 9h+/generation on L4WF (L2WF's last generations
quietly took 5–8h too): each of the ~57 members re-ran its 2 jittered code
variants on the grown window and paid 2 uncached LightGBM combined fits
(~30s each at 550k×158), with the whole-archive "with" fit refit per member
because the fit-cache key was caller-order-sensitive. Three fixes, deployed
mid-run (arms killed + resumed from checkpoints; scp deploy — the server repo
is NOT a git checkout): (1) `harness._marginal_value` now sorts its signals
canonically by fingerprint before fitting, so every rescore member shares ONE
cached whole-archive "with" prediction (the fit key stays ordered; canonical
order is imposed one level up); (2) `_combined_prediction` memoises per-signal
standardised feature COLUMNS (namespaced `("feat", key, window-scope)` entries
in the same per-window fit cache; `_FIT_CACHE_MAX` 128→384, worst ~2 GB,
freed on frontier advance) so X is assembled from shared columns; (3) the
archive rescore skips jitter probes + reference-zoo diagnostics
(`_score_program(include_probes=False, include_reference=False)`) — both were
already computed at admission — and carries the member's admission-time
plateau dock forward (`plateau_penalty_carried` diagnostic, subtracted from
the rescored marginal axis; survives successive rescores). Admission-time
child evaluation is UNCHANGED (full probes). Tests: harness fit-cache tests
updated to the canonical-order contract (+ feature-column counts),
`test_rescore_skips_probes_and_carries_plateau_dock` in
`tests/test_evolution_progressive.py`.
**Root cause found 2026-08-04:** the dominant slowdown was NOT algorithmic —
`lagias-research.slice` (`CPUQuota=200%`, created 2026-08-03 09:02 after a
Hostinger fair-use throttle; see the comment block in `scripts/ladder_lane.sh`)
caps ALL research at 2 absolute cores, and inside it each arm's LightGBM
(`n_jobs=-1` → 8 OpenMP threads × 2 arms) thrashed: an identical 150k×140 fit
measures 23s on the VPS vs 2.7s on the M2, while `OMP_NUM_THREADS=1` restores
~6s. `ladder_lane.sh` now exports `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`
(2 arms × 1 thread = the quota exactly); running supervisors must be restarted
to pick it up (done 2026-08-04 04:04 — both arms resumed on OMP=1). Do NOT
raise the quota without moving research off the box — it protects Lagias
production AND keeps the fair-use monitor quiet. **Temporary lift (user
decision 2026-08-04):** the quota is lifted RUNTIME-ONLY
(`systemctl set-property --runtime … CPUQuota=`; slice file on disk unchanged)
until `L7WF_terra_s0` completes — `/etc/cron.d/restore-quota-l7s0` →
`scripts/restore_quota_after_l7s0.sh` polls its `orchestrator_status.json`
every 10 min, restores `CPUQuota=200%` on ok, logs to
`/root/quota_restore.log` and removes itself (a reboot also restores it). Secondary
factors, in order: wf `reveal-every 1` doubles rescore+signal-warmup events vs
old L4's `reveal-every 2`; organic archive×window growth (fixed above); 232 vs
209 tickers; L5 debate latency (children phase only).

**STRATEGY COMPILER — factors → sellable strategies (built + first results
2026-08-03).** The Selector/Architect route is set aside for the product: a
deterministic, LLM-free compiler (`backtesting/strategy_compiler.py`, design
`docs/research-evolution/FACTOR_TO_STRATEGY_DESIGN.md`) turns a factor book
into positions — blended **RF(0.6)+LightGBM(0.4)** prediction (`rf_gbm`; no
ridge, user decision) → cs z-score → EWMA smoothing (halflife≈horizon) →
sector demean → 1/σ risk scaling → beta neutralisation of the RANKING
(projection against the DEMEANED beta — projecting raw beta then demeaning
reintroduces the market bet; test-caught bug) → dollar-neutral LS leg → net
exposure **from stock picking, NOT an index sleeve** (long leg gross (1+ν)/2
overweights best-ranked names, short leg (1−ν)/2; ν=1 = long-only picking;
user decision) → `max_positions` concentration with entry/exit hysteresis
(retail books 12–20 names; `exit_buffer` rank N·1.5 stops edge churn) →
no-trade band → causal vol targeting. Personas = theme filter
(`select_factors` categories/keywords over the book) + risk params only:
`personas.yaml` (5 product personas + master) and `personas_netlong.yaml`
(master_long30/60/100 + 3 retail netlong). Runner
`scripts/compile_strategy.py --prerun X [--personas-file …]` reuses
`replay_snapshots` (book@gen g trades block g+1) + `_combined_prediction`;
equal-weight buy&hold benchmark row included; `prequential_deployment.py` is
now wf-mode aware (no test/forward frames when `wf_blocks>0`). Tests:
`tests/test_strategy_compiler.py` (9). **First honest results** (net 5bps,
walk-forward): L2WF 2021–26 master MN **0.69** (old construction 0.12);
**master_long30 0.98** @10% vol vs EW-benchmark 0.51 @21% vol (same return,
half the risk); L4 untouched forward 2024–26: MN **1.57**, long30 1.45,
retail_balanced 0.99 (old construction 0.47). 2016–21 segments stay weak
(immature mid-run books). Pending: `strategy_lab` construction grid with
PBO/DSR, L4WF/L5WF books through the compiler, sp100_inject bias bridge.
NOTE: old `data/backtests/sp100_inject` Sharpe 1.40 is NOT a benchmark —
static survivorship-biased universe + factors IC-validated on the full
backtest window (`prerun_inject` look-ahead) + light costs.

**Supervisor walkthrough notebook (built + executed 2026-08-04).**
`notebooks/evolution_walkthrough_for_supervisor.ipynb` — a clone-runnable guided
demo of the evolutionary researcher for the thesis supervisor (only an OpenAI key
in `.env` is needed; ~$1–2/pass, hard `QF_MAX_LLM_COST_USD` ceiling in setup).
Flow: graph snapshot → gap queries/`mechanism_group_specs` → one live seeded
factor with verbatim LLM prompts/replies (via `QF_LLM_TRANSCRIPT_PATH`) →
codegen → 4-axis Pareto scoring explained per-axis (marginal-value decomposition
+ jitter figures, AST-novelty clone demo) → reflection brief → mutation
parent-vs-child → one live generation (`EvolutionLoop`, graphrag, 2 groups
max-mode, `graph_readonly=True`, scratch `out_dir`, metered cost/per-role
table) → real `L4_terra_s0` survivors (code from `state.json`, lineage, rescore
drift plot). Demo data: `quant.config.sp100.yaml` yfinance OHLCV via
`QF_CONFIG_FILE`; fixed book = 8 OHLCV-only formulaic alphas. Runs on a fresh
clone because graph/embeddings/paper index/prebook/L4 artifacts are committed
(fulltext_cache + data/market are not → abstracts-only re-embed for cents +
yfinance download). Generated by a builder script (nbformat); executed outputs
saved in the notebook. Gotcha fixed in-session: a notebook cell calling
`parse_child_response` directly must wrap it in a resample-retry loop (Terra
occasionally emits JSON with unescaped newlines; the loop's
`_parse_and_validate_child` already retries internally).

**GLD HIGH-FREQUENCY L4WF RUN — launched locally overnight 2026-08-03.** The
server ladder's L4WF arm replicated on **single-ticker GLD 10s LOBSTER bars**
(`quant.config.gld_hf.yaml`: lobster provider, tickers [GLD], fundamentals
off; panel 1,409,220 bars × 41 fields 2024-01→2026-06-01, level-5 book) as
prerun `lobster_equity_gld_hf/L4WF_gld_s0` on GPT-5.6 Terra. Same L4WF shape
(20 gens, 8 groups max × 3 demes × 2 children, 12 seeds/group, archive-cap 40,
graphrag + `--graph-readonly`, curation archive, selection-deflation on,
lightgbm) with HF deltas: **forecast horizon 60 bars (=10 min)**;
`--wf-blocks 10 --wf-block-bars 42000` → seed 2024-01→2024-10, phase 1 gens
1–10 → 2025-09, phase 2 gens 11–20 = 10 prequential ~18-day blocks →
2026-06; **no fixed/reference book** (the daily formulaic zoo is meaningless
for one HF name); max-cost $150. Retrieval auto-masks to price/general scope
(no fundamental field in scope → `allowed_scopes={"price","general"}`; the
legacy corpus carries ~259 microstructure/HF papers, and graphrag's gap query
filters mechanisms to those *computable* from the LOBSTER fields). Supporting
seams added (all default-off/byte-identical): **`QF_EXECUTION_LAG_BARS`** in
`backtesting/data_loader.forward_returns` — the label becomes
`close[t+L+h]/close[t+L]−1`, so a signal observed at bar t is only acted on L
bars (run: 3 = 30 s) later, uniformly across IC/marginal/combined-model/
comparison consumers; **`QF_SIGNAL_CACHE_MAX`** env cap for the server-side
signal cache (run: 48 — each HF signal ≈ 12 MB and the local 8 GB Mac cannot
hold the default 512); `LobsterProvider.available_fields` now **sniffs the
on-disk CSV headers** and stops advertising per-level book columns the feed
lacks (GLD has 5 levels, the tier advertised 10 → factors on phantom fields
were all-NaN); `codegen._make_synthetic_panel` gained the per-level
askPrice/askDepth/bidPrice/bidDepth 1–10 columns so per-level factors survive
the smoke test. Ops: `scripts/gld_overnight_supervisor.sh` under
`nohup caffeinate -ims` — relaunch-on-crash (checkpoint resume), RSS watchdog
(4.8 GB → TERM + resume), `QF_USE_MCP=0` (in-process eval; an MCP subprocess
would hold a second panel copy), rc=3 zero-candidates stop / rc=4 budget
stop, then chains `run_model_comparison.py --preruns L4WF_gld_s0 --tickers
GLD --horizon 60 --holding-period 60 --no-downstream` (per-underlying IC +
combined-signal strategy backtest) into `data/comparisons/gld_hf_l4wf/`.
Logs `data/gld_l4wf_{supervisor,run,comparison}.log`. Live smoke
(`smoke_gld`, $0.53) verified graphrag→Terra→codegen→eval→prequential→persist
end-to-end. Known quirk: persist-time `backtest IC@60` is None on a 1-ticker
panel (cross-sectional grid) — the harness/comparison per-underlying ICs are
the meaningful ones.

**Evolutionary researcher — knowledge-graph nested islands replace the QD grid;
4-axis vector (done, 2026-07-21). THIS SUPERSEDES THE TWO BLOCKS BELOW.**
Diversity is no longer maintained by a Quality-Diversity behavior grid but by a
**two-level population hierarchy grounded in the knowledge graph**, and the CORE
vector is back to **4 axes** `("marginal_value", "independence", "parsimony",
"structural_novelty")` — `temporal_robustness` was removed as an axis (the raw
VAL/IS `degradation_ratio` is a reflection diagnostic only, neither scored nor
gated). **Upper level — mechanism groups**: `knowledge/graph_query.
mechanism_group_specs` turns the graph's under-covered Louvain communities (plus
their most paper-supported *computable* gap mechanisms) into `--mechanism-groups N`
reserved groups, each with a focus brief spliced into that group's seeding and
mutation prompts (`loop._group_context`); factors stamp `mechanism_group_id` /
`mechanism` and link back into the graph. Because groups are *defined* by graph
communities, `n_mechanism_groups > 1` **requires `--retrieval graphrag`** and
raises if the graph cannot form that many usable groups — no silent fallback.
**Lower level — demes**: `--demes-per-group M` classic independently-evolving
islands inside each group; `controller.islands` is the flat `N × M` list addressed
by `flat_island(group, deme)` / `coordinates(flat)`, ring migration runs **strictly
within a group**, and the dedup fingerprint is scoped per `(group, deme)` so the
same formula may be rediscovered in a different mechanism context. Each group keeps
its **own reserved Pareto archive** (`controller.group_archives`); the accepted book
is their union, so a strong group cannot dominate a weaker one out of existence.
Groups mix **only** through the explicit low-probability synthesis operators
`cross_group` / `cross_group_splice` (`--p-cross-group`, default 0.10, force-zeroed
when there is one group). CLI: `--mechanism-groups 5 --demes-per-group 3
--children-per-deme 4 --seed-ideas-per-group 6`; `--islands` survives as a hidden
legacy alias meaning demes-per-group. **Removed**: `evolution/qd.py` (`QDArchive`),
`--selection {nsga2,qd}`, `--grid-dims`, `--cell-capacity`, `--depth-gamma`,
`--reuse-omega`, the `trend_reversal`/`signal_speed`/`stress_activation` behavior
descriptors and `tests/test_evolution_qd.py`. `FitnessResult.behavior` remains as a
vestigial empty dict for old-checkpoint round-tripping and has no selection role.
Tests: `tests/test_evolution_mechanism_groups.py` (loop level: graph resolution,
seed/child placement, reserved archives, cross-group operator, focus reaching the
prompt, progressive reveal × groups), `test_evolution_controller.py` (controller
level). Entrypoint drift from this refactor is now caught by
`tests/test_entrypoint_config_kwargs.py`.

**[SUPERSEDED by the block above — `temporal_robustness` was later removed.]
Evolutionary researcher — temporal-degradation gate → 5th Pareto axis (done,
2026-07-16).** The OOS/IS **degradation hard gate** became the `temporal_robustness`
**Pareto axis**, so the CORE vector is now **5 axes** `("marginal_value",
"independence", "temporal_robustness", "parsimony", "structural_novelty")` and the
search gates drop to **coverage + optional cost** (`GateResults.GATES =
("coverage_ok", "deflation_ok", "cost_ok")`; deflation stays a *publish* filter). The
gate was too harsh: a conditioning factor whose standalone IC fluctuates around zero
(|IS IC| just above `min_is_ic`, opposite sign on VAL) was excluded outright — the very
factor class the primary axis exists to protect. As an axis the ratio
`(val_ic·sign(is_ic))/|is_ic|` is **clipped to [−1, 1]** (bounds crowding-distance
normalisation; removes the incentive to game tiny-denominator ratios; symmetric
sign-reversal floor) and traded off against the other objectives; `None` when
`|is_ic| < min_is_ic`. The RAW unclipped ratio survives as the `degradation_ratio`
diagnostic and its reflection advice rule is unchanged. Under progressive reveal the
axis gains meaning: at reveal generations part of VAL was never scored by any earlier
selection, so the ratio there reflects retention on genuinely unseen data (the
`reveal_generation` diagnostic stamp separates them). Legacy state files carrying
`robustness` / `degradation_ok` keys still load (`from_dict` reads only `AXES`/`GATES`);
`EvalParams.gate_degradation` retained (unused). Tests: extended
`test_research_eval_{fitness,harness}.py`,
`test_evolution_{controller,mutation,qd,loop}.py`.

**Evolutionary researcher — 4-axis vector + dev-wide residual IC + progressive
reveal (done, 2026-07-15).** Three overfitting-control corrections. **(1)** The
`robustness` (Probabilistic-Sharpe-Ratio) Pareto axis is **removed** → the CORE
vector is **4 axes** `("marginal_value", "independence", "parsimony",
"structural_novelty")`. The PSR axis was scored on the *same reused* VAL window every
generation, so it could not police the process-level ratchet it was meant to control
(the statistic becomes the optimisation target), folded IC-scale penalties into a
[0,1] probability, and was a near-monotone transform of the primary axis. Its useful
parts — the window-jitter **plateau penalty**, the perturbation-fidelity probe and the
hypothesis **sign bonus** — now fold **onto the `marginal_value` axis** (all IC-scale:
`marginal = raw ΔIC − plateau − perturbation ± sign_bonus`; `sign_bonus` default 0.02→
0.002 for the IC scale). Legacy state files carrying a `robustness` key still load
(`ObjectiveVector.from_dict` reads only `AXES`). **(2)** The **residual-IC**
independence axis now scores on **IS∪VAL** (the orthogonalisation betas still fit on IS
only), quadrupling the effective sample and lowering the axis noise floor without
leaking — factor formulas have no fitted parameters, so only the betas are estimated.
**(3)** New **progressive data reveal** (`--progressive-reveal`, default OFF →
byte-identical baseline): the dev window is revealed block-by-block across generations
(`evolution/progressive.py` `build_schedule` → per-generation `GenerationWindow`), an
**expanding IS + sliding VAL**, so part of each generation's scoring window was never
queried by any earlier selection — *prevention* of the ratchet, not detection. The
window never drops old blocks; a final `--test-frac` tail (default 0.2) is never
revealed (unchanged TEST semantics). Threaded through a calendar-mode split seam
(`is_end`/`val_end` on `mcp/research_{client,server,service}.evaluate_fitness` /
`evaluate_set_fitness`, plus a new `panel_timeline` seam; the signal cache is
window-keyed by the dev frontier). On each reveal the loop: logs a **prequential** OOS
score of the archive on the just-revealed block (`prequential.jsonl` — honest OOS, the
block was never seen), advances the frontier, **re-scores the archive** on the new
window (`controller.rescore_archive`, no `N_trials` billing; drift logged as `rescore`
lineage rows), and frees gate-failing fingerprints for **one** retry
(`controller.release_failed_fingerprints`). Resume-safe (schedule is a pure fn of
(config, index); `failed_fingerprints` persisted; frontier corruption guard). Flags:
`--test-frac --seed-frac --reveal-every --val-blocks`. Tests:
`tests/test_evolution_progressive.py`, updated `test_research_eval_{fitness,harness}`,
`test_evolution_{controller,mutation}` (the `qd` test file is gone with the grid) +
`test_evolution_mechanism_groups.py` (progressive reveal × mechanism groups: the
frontier still advances and each group archive is re-pruned separately).
Progressive reveal is **opt-in** — `--progressive-reveal`, default OFF, so an
un-flagged run is byte-identical to the non-progressive baseline.
**`--final-holdout` (added 2026-07-30, default OFF):** without it the last
revealed block is both selected-on (the post-reveal generations) and ranked-on,
so the final front tilts toward it.  The flag splits the dev remainder into
R+1 blocks, keeps the last hidden from EVERY generation, and reveals it only in
a terminal rescore-only step after the final generation (`schedule[G+1]`:
prequential row + archive rescore at generation G+1, no child fitted after) —
the final front is ranked on data no selection pressure ever queried.
Resume-safe (terminal step idempotent; `_init_progressive` recognises the
post-terminal frontier). Tests in `tests/test_evolution_progressive.py`.
Out of scope (deferred): Thresholdout/
select-guard gate, ε-dominance, GP-arm progressive reveal.

**Evolutionary researcher — residual-IC + regime axes and two-stage curation
(done).** The CORE Pareto vector is now **5 axes**. The **independence** axis is
the candidate's **residual (orthogonalised) IC** (novel predictive content the
book doesn't span — un-saturating, unlike the old Δ-participation-ratio, which
stays selectable via `EvalParams.independence_metric` / `--independence-metric`),
and a new **`regime_independence`** axis rewards *crash-complementarity*: the
marginal ΔIC on the stress bars (`--regime-kind {drawdown,volatility}`,
`--regime-quantile`; stress labelled over IS∪VAL only so it stays leak-free), so a
factor strong exactly where the book is weak is non-dominated and survives even
with worse values elsewhere. The LOCO **marginal-value combiner now defaults to a
nonlinear model** (`gradient_boosting`; `EvalParams.marginal_model` /
`--marginal-model`) so a *conditioning/state* factor — the canonical
low-standalone-IC volatility factor, valuable only via `vol × momentum`-style
interactions — scores a **positive** marginal value (a linear ridge sees only
additive value and scores it ~0; residual-IC independence also can't reward pure
conditioning value). **Two-stage curation (Lever 2)** decouples *what
survives selection* from *what is kept*: `EvolutionController` accumulates a
`kept_pool` of **every gate-passing factor**, and `--curation
{archive,greedy,elastic_net}` (default `archive` = the one-stage Pareto behaviour)
curates that pool **once at the end** (`--n-keep N` optional) — greedy
forward-selection on combined VAL IC or elastic-net stability selection
(`research_eval/curation.py`; MCP `curate_book` reusing the IS-fit/VAL-score dev
panel), so a good factor is no longer discarded merely for being dominated. Both
threaded through the harness/controller/loop/CLI + MCP seam; the Pareto archive
still drives parent selection and marginal scoring. Tests:
`tests/test_research_eval_curation.py`, extended
`test_research_eval_{fitness,harness}.py`, `test_evolution_{controller,loop}.py`.

**Evolutionary researcher — `structural_novelty` axis + genuine canonical-AST
metric (done).** The 5th CORE Pareto axis is now **`structural_novelty`**, which
**replaced the `regime_independence` axis** (crash-complementarity stress lives on
as the QD grid's leak-free `stress_activation` behavior descriptor, not a scored
axis — that descriptor died with the grid in 2026-07-21, so stress-conditional
value is no longer measured anywhere).
`structural_novelty` = the minimum **canonical AST weighted-subtree
distance** to the nearest archive member (falling back to the reference-zoo when
the book is empty); 0 = structural clone (same canonical computation), 1 = no
shared canonical subtree. The measure is a dedicated module
(`research_eval/ast_novelty.py`: `canonical_factor_tree`, `subtree_profile`,
`ast_subtree_similarity/distance`) that **replaced the old whitespace-stripped
`difflib.SequenceMatcher` character proxy** everywhere structural code distance is
used (the axis, `_zoo_dedup`'s `zoo_min_code_distance`, and SET-mode internal
diversity). It parses the `calc` body (best-effort module-body fallback),
canonicalises it (strip locations/docstrings/`pass`; alpha-rename local
variables + params; inline safe straight-line temporaries; **numeric literals →
typed placeholders** so a 20- vs 21-bar window is a *clone*; commutative `Add`/
`Mult` operand reorder — while preserving data-field names, operator/attribute/
function names and `data["close"]` string keys), fingerprints every subtree with
SHA-256 into an immutable multiset profile (`@lru_cache`d by source), and scores
two programs with a **weighted multiset Jaccard** (`w(h)=1+log(1+size(h))`).
Inspired by AlphaAgent's common-subtree originality criterion but a normalised
weighted overlap over *all* subtrees (not the size-biased single largest common
subtree); **not** exact tree-edit distance. Invariant to formatting, comments,
docstrings, factor ids, class names, imports and variable names; sensitive to
changed fields/operators. Faster than the old proxy for archive comparison
(cached profiles + fingerprint-counter Jaccard, no O(n²) character match per
pair). New diagnostics: `novelty_metric`, `novelty_nearest_book_similarity`,
`novelty_candidate_ast_nodes`, `novelty_candidate_unique_subtrees`; the
nearest-book index now points at the original book position even when empty/invalid
codes are skipped. Tests: `tests/test_ast_novelty.py`, extended
`tests/test_research_eval_harness.py`.

**[PARTLY SUPERSEDED — WS2 (the QD grid) was removed 2026-07-21; WS1/WS3/WS4/WS5
still stand.] Evolutionary researcher — QD grid, selection-time deflation,
economic reward, experience memory (done).** Five additions from a competitive-landscape review
(`research_docs/`, gitignored; source attribution log in `research_docs/SOURCES.md`,
which must be updated whenever an external idea is implemented). **(WS1) Deflation
moved off the per-candidate search gate onto a shared *publish* filter**
(`research_eval/publish.py`) run once at materialise over **every** book source
(archive / kept-pool / greedy-elastic / QD elites); it deflates the **combined-book /
marginal (LOCO)** statistic (never standalone `|val_ic|`, which would kill
complementary factors) and prunes by marginal contribution. Search vs publish
eligibility is now explicit (`search_selectable` = gates minus deflation;
`publish_selectable` decided at the end). `--selection-deflation {off=discovery,
on=validation}`. **(WS2 — REMOVED 2026-07-21, replaced by knowledge-graph nested
islands; kept here for the rationale trail.) A Quality-Diversity behavior grid**
(`evolution/qd.py`
`QDArchive`, `--selection {nsga2,qd}`, default nsga2) fills a *diverse* library where
NSGA-II converges: cells keyed by leak-free behavior descriptors — `trend_reversal`
(fade↔momentum), `signal_speed` (slow↔fast), `stress_activation` (3rd axis at
`--grid-dims 3`) — computed by the harness onto `FitnessResult.behavior` (NOT scored;
the 5-axis Pareto is untouched). Each cell is a capped mini-Pareto (`--cell-capacity`)
pruned by the existing crowding/dominance; parent selection samples cell→elite with an
optional **AlphaPROBE depth/parent-reuse bias** (`--depth-gamma --reuse-omega`);
fixed/frozen bin edges (never rebinned). **(WS3) Economic reward folded in, no new
axis**: a `cost_ok` turnover gate via an explicit shared position construction
(`backtesting/positions.py`, not `vector_backtest`) + a **perturbation-fidelity**
robustness probe (`--gate-turnover --cost-rate --perturbation-weight`); all default OFF
so baseline arms are byte-identical. **(WS4) Factor-zoo dedup DIAG** (`_zoo_dedup`):
candidate max-|corr| + code distance + nearest id vs a `--reference-book` (~86 base
factors), diagnostic only. **(WS5) Per-config cross-run experience memory**
(`knowledge/experience.py`, `--memory`): stamps survivor performance AND per-mechanism
attempt/survival tallies (negative evidence → exhaustion detectable), steering next-run
seeding away from exhausted mechanisms and splicing a summary into the teacher. Tests:
`tests/test_research_eval_publish.py`, `test_evolution_qd.py`, `test_knowledge_memory.py`,
extended `test_research_eval_harness.py` + `test_evolution_loop.py`. Notebooks updated
(`notebooks/…_walkthrough.ipynb` §10, `…_live_run.ipynb`).

**GP factor-mining benchmark — non-LLM baseline (done).** A deterministic
**genetic-programming** alpha miner (AutoAlpha spirit: hierarchical evolutionary
mining of formulaic alphas; AlphaGen's "score a combined set" folded in) that serves
as a no-LLM benchmark for the evolutionary researcher. It reuses the LLM arm's
LLM-agnostic machinery **verbatim** — the NSGA-II `EvolutionController` (selection /
Pareto archive / N_trials-deflation / islands / lineage / checkpoint), the
`research_client.evaluate_fitness` scoring seam (in-process under `QF_USE_MCP=0`,
byte-identical), and `persist_archive` — and swaps only the *operator layer*, so it is
apples-to-apples: LLM arm vs GP arm differ in exactly *how children are proposed* over
the same grammar/data/splits/fitness/selection/persistence. New package
`agents/factor_research/gp/`: `grammar.py` (typed expression trees — SERIES/WINDOW/CONST
nodes, an operator table, ramped-half-and-half `random_tree`), `render.py` (tree →
validator-passing `BaseFactor` module; id-independent class name so structural dedup
holds; protected division), `operators.py` (typed subtree crossover + subtree/point/hoist
mutation, depth-capped), `loop.py` (`GPRunConfig` + `GPLoop` — random-tree seeding →
**hierarchical depth-schedule growth** `--depth-schedule 3,5,7` → per-generation
select/operate/render/**smoke+non-degeneracy filter**/admit; `evaluate_program` kept in
sync with `EvolutionLoop`). Entrypoint `run_gp_factor_mining.py` (mirrors
`run_factor_evolution.py`; `scope.activate()` so persist writes into the **prerun**, not
main). **The GP is confined to the base grammar**: its operators come only from the new
explicit `factors.ops.BASE_OPS` tag (never `dir(ops)`), asserted `used_ops(tree) ⊆
BASE_OPS` — the LLM arm may *extend* the grammar (inline helpers / scipy·sklearn·
statsmodels·numpy·pandas via `codegen.validate_code`'s import allow-list), a deliberate
agentic advantage the GP is denied; new ops are opt-in to `BASE_OPS`. Synergy comes free
in SINGLE mode (each candidate scored by its **LOCO marginal contribution to the evolving
Pareto archive** = AlphaGen's combined-set reward). One additive, default-preserving edit
to `persist_archive` (`engine="evolution"`, `model_label=None` kwargs) so GP preruns stamp
honest provenance (`engine=gp`). Output is a standard prerun
(`data/workspaces/<config>/preruns/<name>/`, `source=researcher`, `engine=gp`; main seed
DB untouched) that `run_model_comparison.py --preruns <gp>,<llm>` ingests unchanged — the
non-LLM row of the ablation matrix. SET-mode joint scoring, a canonical single-objective
IC arm, `indneutralize`/cross-sectional terminals, and PCA-directed seeding are documented
extensions. Design: `docs/research-evolution/GP_BENCHMARK.md`; sources logged in
`research_docs/SOURCES.md` (AutoAlpha IJCAI 2020, AlphaGen KDD 2023). Tests:
`tests/test_gp_grammar.py`, `tests/test_gp_loop.py`,
`tests/test_entrypoint_config_kwargs.py`.
**Diversity structure (2026-07-27):** the GP arm has no knowledge graph, so it has
nothing to form mechanism groups *by* — it runs **one** mechanism group containing
`--islands` classic flat demes with the usual within-group ring migration, while the
LLM arm uses the full two-level hierarchy. This is the one deliberate structural
asymmetry between the arms (alongside the grammar advantage) and should be stated
when the ablation matrix is written up. Note also that `--progressive-reveal` is
LLM-arm only; run the LLM arm without it if you want the two arms on identical
splits. The entrypoint's argparse block had drifted twelve kwargs behind
`GPRunConfig` (removed QD / CPCV / regime knobs) and died with a `TypeError` before
any work; fixed 2026-07-27 and now guarded by the entrypoint test above.

**Landing-page example generator (done).** `showcase_pipeline/landing_examples/`
turns real pipeline runs into provenance-stamped marketing artifacts for the
startup spinoff (`../company-brain/marketing/examples/`). `run` drives
Selector→Architect→Statistician attempts and dumps **every** candidate —
approved AND rejected (run_fund persists only approved; the "Likely overfit"
card *is* a rejected candidate) — with full trial history + stat-test details to
`<scope>/landing_examples/candidates/attempt_<nn>.json`; `list` tables each
candidate's harness metrics incl. a **per-strategy CSCV PBO** (`pbo_cscv` over
the return series of the variants the Architect actually tried) + a
deterministic badge (Robust / Worth testing / Likely overfit — fixed thresholds
on DSR/PBO/OOS, never the Statistician LLM's prose); `export --pick` writes per
example: `card.json`, `equity_curve.json` (raw series + polylines pre-scaled to
the landing page's 600×260 SVG; OOS re-based to stitch onto IS), `card.png`,
`behind_the_verdict.md` (idea → factor code → gates → deflation arithmetic),
`chat_transcript.json` (deterministic; optional `--polish-llm` rewrites only the
user turn, flagged), `provenance.json` (config hash, git commit, attempt,
`recompute_match` — the OOS curve is recomputed via the exact `out_of_sample.py`
recipe and cross-checked against the recorded Sharpe). Copy is template-only
with a banned-word compliance guard (hard-fails export). First real export
(prerun `lodestar_demo`: evolution run → 15 factors, 17 strategy attempts):
`robust-meanrev-momentum` (IS 5.16 → OOS 2.07, PBO 0%) vs
`overfit-volume-breakout` (IS 2.01, DSR 0.97, **PBO 74%** — only the CSCV check
catches it). Tests: `tests/test_landing_examples.py`.

**Walk-forward backtest: prerun factor-injection + rich analytics (done).** The
walk-forward harness (`run_backtest.py` + `quant_fund_agent/simulation/`) now has
a second **factor source** besides LLM research: `--factor-source prerun_inject`
replaces the per-meeting LLM Factor-Researcher with a cheap, deterministic draw
from one or more **preruns** — `--inject-preruns a,b` (under `--inject-config`,
default the derived data-config scope e.g. `yfinance_equity_sp100`) pools their
RESEARCHER factors (union, dedup), shuffles once with the run seed, pre-filters to
those **computable on the live panel** (mirrors the comparison harness's gating),
and injects `--factors-per-meeting` (default 2) onto the run catalog on each
research-due meeting (seeds always available; the pool is exhausted once every
factor has been injected). Factor *code* already lives in the shared
`factors/researcher/` package so signals compute with no extra wiring; only the
catalog *records* are moved (`simulation/factor_injection.py`). The harness is now
**config-driven for API providers** — `run_backtest.py --config quant.config.<x>.yaml`
exports `QF_CONFIG_FILE` before the panel loads, so a yfinance/FMP universe + date
range (e.g. the static, survivorship-biased `sp100` list since 2016) drives the
whole run; `quant.config.sp100.yaml` is the ready-made yfinance S&P100 config.
**Analytics** are now presentation-ready: the execution layer tracks per-bar gross
exposure + names held, `BacktestResults` adds a NAV path / % invested / drawdown to
`equity.csv` + `fund_metrics.json`, and `simulation/report.py` renders a `report/`
folder of figures (NAV+drawdown, cumulative return, percent invested & names held,
rolling Sharpe, monthly returns, per-strategy attribution, catalog/strategy growth)
plus a `report.md` KPI table. All meeting cadences (`--research-every`,
`--strategy-every`, `--pm-every`, `--grid-freq`) are independent, so "monthly
everything" is `--grid-freq 1M --research-every 1M --strategy-every 1M --pm-every 1M`.
Tests: `tests/test_simulation_injection.py`.

**Research-LLM comparison (done; extended with analytics + speed/reliability).**
Named factor-research *preruns* (`run_factor_research.py --name <id> --model <llm>
[--llm-provider <p>] --dedup-scope prerun`) mine N factors with a chosen research
model into a self-contained `data/factors/preruns/<name>/`. `run_model_comparison.py`
then compares several preruns' factor sets on **four** axes — **single-factor IC**
(by default a **per-underlying time-series IC** — one Pearson correlation per asset
between a factor's value vector and that asset's *own* forward-return vector,
aggregated as a valid-observation-weighted mean, so it is well defined for a
single ticker and has no cross-section;
`--fit-standardize cross_sectional` switches back to cross-sectional IC),
**factor analytics** (LLM-free:
*diversity/redundancy* — signal correlation, effective # of independent factors via the
participation ratio, cluster count; and *deflation/importance* — best |IC| haircut for
the number of factors tried, plus LASSO/GBM feature importance & sparsity), **ML-combined
signal → per-underlying vectorised backtest** (each catalog model + ensemble combines the
factors into ONE predicted signal, fit on IS; that combined signal is backtested *not*
cross-sectionally but as a standalone directional bet per underlying, OOS;
`comparison/vector_backtest.py` + `bruteforce.py`). To avoid the Sharpe bias from
overlapping forward returns (a single `position × h-bar forward return` row overlaps
`h−1` bars with its neighbour → annualised return inflated ~`h`× and Sharpe ~`√h`×),
each bar's target is held as a **staggered "tranche" book** — the live position is the
rolling mean of the last `holding_period` targets (`1/h` capital layered in per bar) —
**marked to market on the 1-bar forward return** (`book[t] × forward_return(t→t+1)`),
which is the same non-overlapping convention the deployed `strategy_backtester` uses, so
research and deployment can't drift. `--holding-period` sets the tranche length
(default = `--horizon`; the forecast/IC horizon stays `--horizon` regardless). Also
**downstream agents** (the full Selector→Architect→Statistician→PM
fund, single-pass OOS) — emitting presentation-ready figures, CSV/JSON tables, a
`report.md` and a `comparison.ipynb` under `data/comparisons/<id>/`
(`quant_fund_agent/comparison/`). The backtest's modelling choices are all CLI args (both
sides implemented): `--fit-scope {pooled,per_underlying}` (default **pooled** — ONE model
across all underlyings; `per_underlying` fits a separate model per name, for heterogeneous
data-rich universes like the LOBSTER ETFs), `--position-mode {threshold,sign,continuous}`
(default threshold band), `--position-zscore {expanding,full,rolling,none}`,
`--aggregation {portfolio,per_underlying}`, `--fit-standardize {per_underlying,cross_sectional}`
(default per-underlying time-series z-score on IS stats). **`--fit-standardize` now governs
the WHOLE comparison**: at the default `per_underlying` the IC track, the analytics
diversity/importance fits (`comparison/analytics.py` `_feature_matrix`) and the brute-force
fit + combined-signal IC are *all* per-underlying (shared
`comparison/standardize.per_underlying_zscore`) — **no cross-section anywhere** — so every
track is meaningful for a single ticker; `cross_sectional` restores the legacy across-tickers
z-score + IC. `--importance-top-n` (default 10) caps factors per (prerun, model) in the
importance table; set it high to keep the full per-factor vector. Everything except the
downstream track and `--research` is LLM-free.
**Universe + split selection:** beyond `--n-tickers N` (a count) you can name the exact
underlyings with `--tickers AAPL,MSFT,CORN` (overrides the count), and beyond `--oos-ratio`
(a tail fraction) you can split IS/OOS by the *calendar* with `--train-months`/`--oos-months`
— each a comma list of months/dates (`2024-06,2024-07` / `2024-06-15`) or an inclusive range
(`2024-06:2024-08`); they must be disjoint, the panel is restricted to their union so every
track scores those months, and the split seam is `ComparisonConfig.split_masks(index)` (used
by the brute-force/vector-backtest tracks; the LLM downstream track keeps its ratio split).
**Faster + crash-safe:** `--max-bars N` (default 20000 under `--fast`)
uniformly strides the panel so *every* track is fast and the brute-force OOM is gone;
the harness **checkpoints tables+figures after each track** (writing `status.json`),
so an interrupted run never loses completed tracks. Factors needing fields the current
data lacks are filtered (and reported), so it runs on today's LOBSTER sample now and
re-runs unchanged once full LOBSTER / FMP data is downloaded.
**Rolling-window sweep (`run_rolling_comparison.py` + `comparison/rolling.py`).** Automates
the comparison **per ticker over a rolling IS/OOS month window** (default 2 IS months + the
next OOS month, stepping one month forward so the prior OOS month becomes the second IS
month) for every ticker under `ticker_data/`, comparing the preruns **per underlying**. Each
(ticker, window) runs in its **own subprocess** — the OS reclaims its memory between runs, so
the large intraday panel never accumulates (no OOM); the sweep is **resumable** (windows whose
`status.json` is all-`ok` are skipped) and robust (a failed run is logged; the sweep
continues). It then **aggregates** every run into `data/comparisons/<batch>/`:
`combined/{bruteforce,importance,diversity,ic}_all.csv` (tagged `ticker,oos_month,is_window`),
per-ticker `importance_over_months__<prerun>__<model>.csv` + heatmaps (the **most important
features over the OOS months**) and `performance_<metric>__<model>.png`, plus **cross-ticker
figures** (`cross_ticker/`, built by `run_rolling_comparison.build_cross_ticker_figures`): per
factor set, OOS Sharpe / combined-signal OOS IC / OOS÷IS-Sharpe ratio over the OOS months with
**one coloured line per ETF** (mean across models), and one **mean-OOS-Sharpe vs average-daily-
volume** scatter (one dot per ETF, coloured by `TICKER_SECTORS` asset class; ADV from per-bar
`trdLiq`), plus `summary.md` and `manifest.json`. Tests: `tests/test_rolling_comparison.py`.

**Researcher data-scope gating — done.** The setup wizard now asks which data
the Factor Researcher may use, and that choice flows all the way through to the
factors it invents. For LOBSTER it asks the **order-book level** — `lobster_level`
in `quant.config.yaml`: **2** = order book only (mid-derived OHLC, traded volume,
spread/depth/imbalance + per-level book), **3** (default) adds the
message-stream fields that can't be reconstructed from the book (`trade`,
`orderFlow`, `hidden`, `auction`, `effSpread`, `trdLiq`, `ofLiq`,
`nbEvents/nbHidden/nbTrades`). For the API vendors it asks whether to use
**fundamental** data (the existing `fundamentals` flag; equity FMP/AlphaVantage
only). The split lives in `data/tiers.py` (`lobster_fields_for_level`,
`LOBSTER_L2/L3_MICRO_FIELDS`); `LobsterProvider.available_fields()` honors the
level so gating is consistent *everywhere* (seed factors, comparison harness,
catalog), with **level 3 the default → existing configs unchanged**.
`data.usable_fields(settings)` is the single run-scope field set (provider caps −
non-OHLCV when fundamentals off); `pipeline.run_research_session` computes it and
threads it onto `FactorResearcherState.allowed_fields`. The brainstorm/codegen
DATA CONTEXT is built per-run by `agents/factor_research/prompts.build_data_context`
(only in-scope fields, empty sections dropped, LOBSTER-vs-generic intro), and
`filter_and_persist` drops any factor whose declared `inputs` fall outside the
scope. Tests: `tests/test_factor_research_data_scope.py`.

**Position construction — configurable, default by data type (done).** How a
deployed strategy's composite signal becomes positions is now a regime, resolved
once per run and **stamped on the StrategySpec/StrategyRecord** so the Architect
IS-fit, the Statistician OOS test and live/PM all reproduce the same book:
**`cross_sectional`** (the prior behaviour — cross-sectionally rank the signal,
trade the top `max_positions`, scale to a dollar-neutral long/short book) vs
**`per_underlying`** (standardise each name over its *own* history, go
long/flat/short by a boundary — `position_mode` `threshold`/`sign`/`continuous`,
`position_zscore_basis` default causal `expanding` — then size each active name
to **`1/max_positions`** equal weight; directional, net market exposure, not
dollar-neutral). Default is **provider-keyed**: `per_underlying` for LOBSTER
(heterogeneous, data-rich ETFs), `cross_sectional` for the API/stock universes;
override with `run_fund.py --position-construction {auto,cross_sectional,per_underlying}`
or `QF_POSITION_CONSTRUCTION`. The two boundary primitives (`zscore_over_time`,
`directional_positions`) live in `backtesting/positions.py` and are **shared**
with the research comparison track (`comparison/vector_backtest.py`) so research
and deployment can't drift. The model catalog is unchanged (linear gives factor
weights; trees/boosting stay available). Equal-weight `1/max_positions` sizing is
the v1; inverse-vol / other sizing is a documented future extension. Tests:
`tests/test_position_construction.py`.

**Per-factor prediction horizons — done.** A factor no longer borrows a single
externally-passed horizon; it carries its own. `BaseFactor.prediction_horizon`
(canonical, validated like `inputs`) + optional `suggested_horizons` (in *bars*)
are stamped onto `FactorRecord` at persist (the dual of `required_inputs`). The
Factor Researcher now *chooses* the horizon: the brainstorm/codegen prompts get a
**bar-size-aware HORIZON CONTRACT** (seconds-per-bar inferred from the panel index
via `data/frequency._median_bar_seconds`, surfaced by `build_data_context(...,
seconds_per_bar)` — no hardcoded "10s"), and `codegen` requires a positive-int
`prediction_horizon ≤ MAX_PREDICTION_HORIZON`. Existing factors were backfilled to
**constant 6** (`FactorDatabase.backfill_prediction_horizons`,
`scripts/backfill_horizons.py`; idempotent via a `metadata` sentinel; a
`data_driven` best-`|IC-IR|` mode also exists). **IC** is now anchored at each
factor's *own* horizon: `engine.backtest_factor(..., factor_horizon=)` unions it
into the grid and headlines it (`factor_horizon=None` is byte-identical to the old
path), and the comparison IC track emits `ic_own`/`icir_own`/`horizon_own` +
`mean_abs_ic_own` alongside the shared grid. **Combined signal:** the brute-force
track derives its forecast horizon as the **mode** of the constituent factors'
horizons (`ComparisonConfig.resolve_target_horizon`, `combined_horizon_agg ∈
{mode,median,max,min,explicit}`; `run_model_comparison.py --horizon` is now an
*override* → `explicit`, `--combined-horizon-agg` sets the default); the
**downstream Architect** is shown each factor's `prediction_horizon` and **chooses**
`StrategySpec.target_horizon` (was config-fixed), persisting it across revise
iterations. Tests: `tests/test_factor_horizon_codegen.py`,
`tests/test_factor_horizon_backfill.py`, `tests/test_horizon_pipeline.py`,
extended `tests/test_comparison.py`.

**Pluggable data layer (Phases 0–6 done — milestone complete).** The fund no
longer requires the author's LOBSTER CSVs: any panel load goes through
`quant_fund_agent/data/` (provider abstraction + parquet cache + capability
gating + calendar-aware annualization). Providers: `lobster` (local CSVs),
`yfinance` (key-free daily), `fmp`, `alphavantage` — the three API vendors are
**multi-asset (equity / crypto / fx)** via canonical `BASE-QUOTE` symbols
(`data/symbols.py`); crypto annualizes at 365 (inferred from weekend bars). Run
`python -m quant_fund_agent.setup` to pick a provider/asset-class/universe/
timespan and write `quant.config.yaml`, or
`python -m quant_fund_agent.setup --assist "<plain-English description>"` to have
an LLM draft a config you confirm (`setup_assist.py`; validated to legal values,
falls back to the deterministic wizard with no LLM key). See
`docs/data-layer/ROADMAP.md`.

**Non-OHLCV data — Stage 7 done (fundamentals + estimates/events).** Factors can
now read availability-stamped fundamentals (`sector`/`peRatio`/`roe`/`revenue`/…),
analyst estimates (`epsEstimate`/`revenueEstimate`) and earnings events
(`epsSurprise`) from FMP + AlphaVantage. Point-in-time is enforced in the data
layer (`data/fundamentals.py`: stamp at filing/`reportedDate` else fiscal-end +
lag → forward-fill onto the daily index with a staleness cap; `_truncate_as_of`
then keeps look-ahead out). Canonical field vocabulary + per-vendor normalization
(`data/fields.py`), enriched `fundamental` tier + new `estimates`/`events` tiers
(gating unchanged), a quarterly-TTL record cache (`cache.py::cached_records`), an
additive `ApiProvider._fetch_fundamentals` hook, the `indneutralize` wide-frame
fix, and example factors in `factors/fundamentals/`. Equity-only; `QF_FUNDAMENTALS=0`
to opt out. Live: AV's free tier delivers these; FMP's free tier serves only
`profile` and paywalls the rest (factors degrade, never crash). **Next (designed,
not built):** sentiment + macro — see `docs/data-layer/FUNDAMENTAL_AND_ALT_DATA.md`.

**FMP Premium archive — local, survivorship-bias-free S&P 500 + Nasdaq-100 back to
2004 (DONE, downloaded 2026-07-27).** The archive is **built and on disk**: 1 107
tickers, ~37.6k calls, ~3.5 GB fetched → **1.0 GB parquet**, ~55 min at 600
calls/min; 20 832 manifest rows (19 375 ok / 1 456 empty / 1 restricted). The
headline result is the **point-in-time constituent coverage** — the share of each
date's *actual* index members the archive can price: **100 % from 2020, 98 % from
2016, 93 % 2012, 89 % 2008, 84 % 2004** (Nasdaq-100 tracks it within ~2 pts). So
the panel is effectively survivorship-bias-free from ~2016 and 84–93 % complete
before that; the residual is a **vendor limit, not a resolution bug** — FMP has no
security at all for Bear Stearns/AT&T Wireless/Countrywide/Cephalon/Andrew/BEA
(`search-symbol` + `search-name` both empty), and `ABKFQ` maps only to `AMBC`, the
*post-bankruptcy* entity, which must NOT be spliced onto the old series. By exit
decade: 57 % of names that left 2004–09 resolve, 61 % 2010–14, 78 % 2015–19,
**100 %** for 2020+/still-members. The FMP and free tables resolve almost
identically (837/955 = 87.6 % vs 845/979 = 86.3 %) and disagree mostly on *ticker
vintage* (BNY/BK, BALL/BLL, BKR/BHGE, AA/ARNC, AMBC/ABKFQ), not membership — so
the download ran over the **union of both**, making the choice of masking table a
config change rather than a re-download. Verified end-to-end: `load_panel` through
`fmp_archive` gives 5 676 bars × 837 tickers (53 % dense = the PIT mask working),
ATVI dark after 2023-10-16 (Microsoft acquisition), `peRatio` stepping 90×
(quarterly), `marketCap` daily — **~160 s for a 6-field 22-year load** (a known
cost; cache assembled record frames if repeated loads bite). With a Premium key
the free path's two gaps close. `quant_fund_agent/data/fmp_ingest/` is a one-time,
**resumable** bulk downloader (`scripts/fmp_bulk_download.py`): an endpoint
**registry** (`endpoints.py`), a rate-limited threaded `client.py` that classifies
**402 plan-restriction (terminal, never retried) vs 429 (back off) vs transient**,
a `capabilities.py` **probe** (`--probe` → `capabilities.json`; FMP gates by
endpoint, by *parameter value* — `period=quarter`, a numeric `limit` cap — and by
**individual symbol**, which is why delisted tickers 402 on lower plans), a
`store.py` archive with an append-only manifest journal compacted at the end (one
unit of work = `(endpoint, period, symbol)` = one parquet + one manifest row, so a
killed run re-enters exactly where it stopped), `symbols.py` delisted-name
resolution (literal ticker → `.`/`-` variants → `symbol-change` chain → stripped
bankruptcy suffix; files keyed by the **membership** ticker) and `download.py`
orchestration. ~1 300 tickers × ~22–30 calls ≈ 30–40k calls ≈ 5 GB at 600/min
(~1–1.5 h). Membership is now **FMP-native**: `membership.FmpSource` is
implemented (was a stub) and `scripts/build_fmp_membership.py` backward-walks
`historical-{sp500,nasdaq}-constituent` into the **existing canonical schema**
(so `membership.py`/`resolve_universe`/the per-bar mask are unchanged), audited
(count band, no overlaps) and reconciled **per year** against the preserved free
reconstruction `sp500_public.csv` — itself rebuilt back to 2004, which alone
recovers **145** members that left before 2010 (Bear Stearns, Ambac, AT&T
Wireless…). Consumption is `data/providers/fmp_archive.py` (`provider:
fmp_archive`, `quant.config.{fmp_sp500,nasdaq100}.yaml`): reads the archive
**offline**, filters to requested `fields` *before* materialising (the full panel
would be GBs), and — the real PIT upgrade — `ratios`/`key-metrics`/
`financial-growth` carry no filing date, so they **inherit the matching income
statement's actual `filingDate`** joined on `(fiscalYear, period)` (earliest
filing wins, so a restatement can't hide a value that was public) instead of the
flat 60-day lag. The canonical vocabulary grew 17 → **~130 fields** via one table,
`fields.ARCHIVE_FIELD_SPECS`, which now drives the tier sets, the per-endpoint
normalisation maps **and** the researcher's DATA CONTEXT prose so they can't
drift. Shared interval algebra (`coalesce_spells`, `members_from_spells`,
`audit_spells`, `compare_spells`, `normalize_ticker`) moved into
`data/membership.py`; `build_sp500_membership.py` now imports it (verified
byte-identical: 850 spells / 834 tickers / Jaccard 0.9111). **Units fix:** the
legacy map had canonical `freeCashFlow` pointing at FMP's `freeCashFlowPerShare`;
`freeCashFlow` is now absolute USD and `freeCashFlowPerShare` is its own field.
Not PIT-backfilled by design: `profile` and `shares-float` are *current*
snapshots (the PIT share count is `sharesOutstanding` =
`weightedAverageShsOut`). Unadjusted prices + dividends + splits are archived so a
PIT adjustment factor can be rebuilt later. Tests (offline, synthetic payloads):
`tests/test_fmp_ingest.py`, `tests/test_fmp_archive_provider.py`. Docs:
`docs/data-layer/FMP_PREMIUM_ARCHIVE.md`.

**Survivorship-bias-free S&P 500 — point-in-time membership (done).** A static
ticker list over a 2010→today backtest over-represents survivors. `DataSettings.
membership="sp500"` (or `QF_MEMBERSHIP`) turns the universe **time-varying**:
`resolve_universe` returns the **union of every name ever an S&P 500 member** in
`[start,end]` (834 distinct tickers since 2010 vs ~503 today) so names that later
left still load, and `data/panel.load_panel` applies a **per-bar boolean mask**
once at load (`data/membership.py::apply_membership_mask`) — `NaN`-ing every
`(date,ticker)` cell where the ticker wasn't a constituent. Because it's the single
panel-load seam, research, the Architect/Statistician, the walk-forward trade loop
and the comparison harness are **all** survivorship-correct with no per-loop change
(cross-sectional ops skip the NaNs; the cutoff `_truncate_as_of` is orthogonal).
The canonical interval table `data/universes/membership/sp500.csv`
(`ticker,name,start_date,end_date,add_reason,remove_reason,source`; end exclusive)
is reconstructed from **free public sources** by `scripts/build_sp500_membership.py`
— **primary** GitHub `fja05680/sp500` (date→full-set series, forward run-length
scan into spells) cross-checked against **Wikipedia** (current + "Selected changes"
backward-walk, month-end Jaccard 0.91 mean rising 0.84→0.99; reasons + rename
detection). Renames (FB→META, …) coalesce into one continuous spell under the
current ticker. Audited (month-end count 497–506; no overlaps; TSLA absent-2015/
present-2022, ATVI present-2016/gone-2024) in-build and in `tests/test_membership.py`.
Build is idempotent + offline-replayable (dated raw snapshots under
`membership/sources/`, `MANIFEST.json`). **Free path is tickers-only** — yfinance
can't serve most delisted names (≈82% sampled coverage, the residual bias a premium
`CrspSource`/`FmpSource` closes via the `MembershipSource` seam). The static
`sp100.txt` is untouched. See `docs/data-layer/SP500_MEMBERSHIP.md`.

**Strict modularisation by (config, prerun) — done.** Researched factors and the
strategies built from them are isolated per **(data config, research-LLM prerun)**
under `data/workspaces/<config>/preruns/<prerun>/` (factors, strategies, returns,
fitted model artifacts, portfolio, showcase) via a single layout seam
(`quant_fund_agent/workspace.py`: `Scope`/`Book`, consumed by `run_fund.py`,
`run_factor_research.py`, the simulator and the merge tool). Returns and `.joblib`
artifacts are now scoped through the `STRATEGY_RETURNS_DIR` / `MODEL_ARTIFACT_DIR`
env seam (read live, inherited across the MCP subprocess — same pattern as
`FACTOR_DB_PATH`), so parallel runs never collide. The canonical *main* factor
library `data/factors/factor_db.json` holds **only seed/formulaic alphas** and is
never written by research; every factor/strategy is stamped with a `provenance`
(config hash + scope). `run_merge.py` (`quant_fund_agent/merge.py`) composes a
separate *active book* under `data/books/<name>/` by pooling chosen scopes'
factors + strategies for the PM — main is never mutated, cross-config pooling is
warned not blocked. `scripts/migrate_main_seed_only.py` splits the legacy mixed
main DB into seed-only main + a preserved `legacy` scope (backed up, idempotent,
`--dry-run`). `factors/preruns.py` is now a compatibility shim over `Scope`.

**LOBSTER ingestion — built & validated against the live site.**
`quant_fund_agent/data/lobster_ingest/` converts LOBSTER exports into
`ticker_data/{TICKER}/bin{YYYYMM}.csv` (`converter.py` + `scripts/convert_lobster.py`):
levels-agnostic, streams **one day at a time** (RAM-bounded), clips to the regular
session → 2340 bars/day, and **auto-detects the product** — *raw* message+orderbook
(19 aggregate cols) vs *sampled* 10s book (book cols only, flow cols empty). **Both
also emit per-level book columns** `askPrice{i}`/`askDepth{i}`/`bidPrice{i}`/
`bidDepth{i}` (`4·NumLevels` of them, bar-mean of price + displayed depth at each
level; `converter.output_columns(N)`/`level_field_names(N)`). The loader discovers
them from the CSV header and forwards each as its own panel field; the
microstructure tier advertises levels 1–10 so factors using them aren't gated out.
Derived columns are **reconstructed** from LOBSTER docs (original 2019 aggregator
isn't in the repo; defs + drift-warning in `docs/lobster-ingestion/CONVERSION_SPEC.md`).
`converter.convert_archive` streams a `.7z` day-by-day so a 2-yr raw archive never
unzips in full. The autonomous pull (`orchestrator.py` + `lobster_portal.py` +
`scripts/run_lobster_ingest.py`) drives the real portal: one-time `--recon` login
(persisted cookie `storageState`), `--place-raw` (single `requestdata.php` form,
one request/ticker), then `--ingest` reads finished orders off `mydata.php`,
stream-downloads each `.7z`, converts, and **deletes it** — resumable via
`orders_done.json`, disk-governed. Live-verified: downloaded+converted a CORN
level-10 order (500 days); a full **level-5 CORN pull (16 archives, 499 trading
days, 2024-06→2026-05, ~1.16M bars)** is converted into `ticker_data/CORN/` with
the 19 aggregate + 20 per-level columns (3 fully-empty thin-ETF days → `_empty_day`
NaN grids). **Key product finding:** only the **single `requestdata.php` form
yields RAW message+orderbook (all 19 columns)**; the **bulk
`requestBulkData.php` form delivers an order book only (no message stream → no
flow/trade columns)** regardless of `interval`. Download names: `…_LEVEL.7z` (raw)
vs `…_LEVEL_INTERVAL.7z` (sampled). Tests:
`tests/test_lobster_{converter,orchestrator}.py`. Scratch (`data/lobster_raw/`,
cookie `.lobster_state.json`) is gitignored. See `docs/lobster-ingestion/`.

## Roadmap
- Longer backtests: simulate weekly researcher updates and trade over an
  extended period (the `pipeline.py` stages are built to be called on a
  schedule).
- Deepen the sophistication of every agent.

## Conventions
- Use Anthropic's MCP for agent/tool interactions.

## Python Environment
Always use the local virtual environment interpreter directly.
- Run python scripts: `./venv/bin/python path/to/script.py`
- Run tests: `./venv/bin/pytest`
- Install packages: `./venv/bin/pip install <package>`

Always update the global README.md and claude.md files after  significant changes to the last statusses mentioned in there. 
