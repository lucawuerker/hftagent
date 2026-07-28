# Final Comparison Run — Plan & Decision Record

**Status: DECISIONS SIGNED OFF (2026-07-27). Implementation in progress in this session; this
document is the handoff for the next agent session that executes the runs.**

This is the master plan for the thesis comparison run of the evolutionary factor researcher
(`run_factor_evolution.py` + `agents/factor_research/evolution/`). It records (A) every decision
agreed with Luca, (B) the code changes being made before any credits are spent, (C) the exact run
matrix with cost/time estimates, (D) the smaller verification tests that must pass before each
stage, and (E) the deliverables (results + walkthrough notebook).

---

## A. Signed-off decisions (2026-07-27)

### A1. Models (6 arms)

| Arm | Model | API / route | Price in/out per MTok | Est. $/full run (debate on) |
|---|---|---|---|---|
| OpenAI frontier | **GPT-5.6 Sol** | OpenAI API ($2,500 credit) | $5 / $30 | ~$80 |
| OpenAI cheap | **GPT-5.6 Luna** | OpenAI API | $1 / $6 | ~$16 |
| Anthropic mid | **Claude Sonnet 5** | **Amazon Bedrock** (`anthropic.claude-sonnet-5`) → $10k AWS credits | $3/$15 (intro $2/$10 til 2026-08-31 on 1P; Bedrock partner-priced — verify) | ~$30 |
| Anthropic frontier | **Claude Fable 5** | Amazon Bedrock (`anthropic.claude-fable-5`) | $10 / $50 | ~$150 |
| Workhorse (ladder) | **Claude Opus 5** | Amazon Bedrock (`anthropic.claude-opus-5`) | $5 / $25 | ~$75 |
| Meta | **Muse Spark 1.1** | Meta Model API (public preview; $1,000 Meta credit; verify access + OpenAI-compat) | $1.25 / $4.25 | ~$17 |

- Claude routing: **Bedrock for bulk** (billed as native AWS usage → AWS credits apply; Claude
  Platform on AWS / Marketplace does NOT qualify for promo credits — do not use it). $500 direct
  Anthropic credit is the spillover/fallback.
- All roles (hypothesis/debate/codegen) use the **same model within an arm** — clean attribution.
- MUST probe each provider with a 1-call smoke test before any run (see D1).

### A2. Experimental design

- **Ablation ladder on ONE workhorse model (Claude Opus 5), then model sweep at the full config,
  2 seeds each.** Walk-forward final validation + secondary-universe rerun on the winner only.
- Ladder (all with progressive reveal ON, identical data/splits/seeds):
  - **L0** GP baseline (`run_gp_factor_mining.py`, no LLM, 1 group × flat demes — known asymmetry, stated in write-up)
  - **L1** oneshot baseline (`run_factor_research.py`)
  - **L2** evolution, `--retrieval none` (LLM-knowledge/creative only)
  - **L3** evolution + `--retrieval rag`
  - **L4** evolution + `--retrieval graphrag` + mechanism groups
  - **L5** = L4 + `--debate on`  ← **the full config** (debate A/B = L4 vs L5)
  - **L6** = L5 with `--evolution-unit set`
  - **L7** = L5 + `--memory` (memory A/B = L5 vs L7; requires per-arm memory keying fix)
- Model sweep: **L5 config × {Sol, Luna, Sonnet 5, Fable 5, Muse Spark 1.1}** × 2 seeds
  (Opus 5's L5 ladder rows double as its sweep entry).
- Final: `--walk-forward` (~3 folds) on the winning config; secondary-universe rerun (S&P 500 PIT)
  of the winning config for robustness.

### A3. Data

- **Primary universe: Nasdaq-100 point-in-time** (`membership: nasdaq100`, FMP archive,
  survivorship-bias-free), **start 2010-01-01**.
- **The last two years are reserved for forward testing and never enter any run**: panel
  `end = 2026-07-27 minus 2y = 2024-07-27`. Enforced by config file (`quant.config.<run>.yaml`
  end date), not by flags — nothing downstream can see past it.
- Within the visible panel, progressive reveal's `--test-frac 0.2` tail is additionally held out
  (touch-once TEST), so the honest evaluation chain is: dev (revealed progressively) → TEST tail
  (touch once) → 2-year forward reserve (never touched until live forward test).
- Full PIT cross-section (no `--n-tickers` cap; the cap default of 15 is a bug for this purpose —
  fixed to allow 0 = full universe). Secondary universe for robustness: S&P 500 PIT, same dates.
- Marginal model: keep `gradient_boosting` if the FMP timing probe (D2) shows acceptable
  per-candidate cost with the new speedups; else fall back per probe results (documented in run log).

### A4. Survival / archive policy (changes from current behaviour)

- Keep Pareto front-1 archive semantics for *search*, but:
  - **Per-group archive cap** (~25, crowding-distance cull; evictions logged to lineage).
  - **End-of-run curation ON**: `--curation greedy` (book = greedy forward-selection on combined
    VAL IC), `kept_pool` still saves every gate-passer (= "all factors of all runs are saved").
  - **Publish-time deflation ON**: `--selection-deflation on`.
  - **Fail-open bugs fixed**: curation/publish failures must not silently persist the unfiltered book.
- NOT adding a min-marginal admission threshold (protects conditioning factors).

### A5. Progressive reveal (ON for every evolution arm — hard requirement)

- `--progressive-reveal --reveal-every 2` (~10 reveals over 20 generations).
- Rescore-cost speedups: shared/full-book LOCO fit cache (~50% of rescore cost), plus
  diagnostic-work skips where semantics allow.
- Drift tracking upgrades: reveal-index + window-frontier stamps on `prequential.jsonl` and
  lineage `rescore` rows; rescore failures logged; archive prunes on rescore logged;
  `lineage.jsonl` resume-truncation bug fixed; prequential duplicate-row-on-resume deduped.
- Per-generation book-quality log (`gen_quality.jsonl`) so the *evolutionary* effect across
  generations is separable from the *data-reveal* effect across reveals.

### A6. Knowledge base

- **+~1000 papers** via new query blocks in the existing `scripts/populate_papers.py`
  (~fundamentals/quality/accruals/analyst + pure math/ML/signal-processing/statistics),
  descriptions with **gpt-4o-mini** (cost saving, as before).
- **`data_scope` tagging** at harvest (per query block: `fundamental` vs `price` vs `general`),
  plus a retrieval mask so fundamentals papers are filtered out of OHLCV-only arms
  (satisfies "flag them so OHLC-only agents don't use them").
- Re-embed once with `QF_EMBEDDER=openai` (`text-embedding-3-small`, ~$5), then build the
  knowledge graph once (`scripts/build_knowledge_graph.py`, gpt-4o-mini, ~1-2h) —
  **the graph does not exist yet and graphrag arms crash without it.**
- RAG brainstorm paper-text token cap added (currently uncapped → 60k-char papers inline).

### A7. Other agreed points from the review

- **Cost/token tracking**: provider-aware meter at the `make_chat_llm` choke point; per-role and
  per-run totals persisted to `run_config.json` / `summary()` / `manifest.json`; hard
  `--max-cost-usd` ceiling per run. This is the per-LLM cost overview for the thesis.
- **Provenance fix**: per-role models + providers stamped into `run_config.json` and manifest.
- **Creative ideas**: small `--creative-frac` (default 0.0 → byte-identical; run at ~0.1) mixing
  explicitly-ungrounded, novelty-encouraged hypotheses into grounded arms; zero-citation ideas
  already survive verification (`source_paper_ids=[]`).
- **Mechanism briefs**: hypothesis prompt relaxed — mechanism may be 2–4 sentences and may be
  mathematical/statistical/behavioural, not only economic (Hawkes-style reasoning welcome).
- **One prediction horizon** for the thesis runs (`--prediction-horizon-mode fixed`, horizon 6);
  multi-horizon later.
- **Memory ablation**: memory store keyed per (config, prerun-arm) so L7 doesn't contaminate
  other arms.
- **Meta-agent for steered graph-expansion runs**: deferred (future work section of thesis).
- **Live web search during runs**: deferred; corpus expansion via arXiv harvester covers
  "expand beyond current dataset" for now.
- 4-axis Pareto vector verified at source (`AXES = (marginal_value, independence, parsimony,
  structural_novelty)`; no temporal/CPCV element anywhere in selection). 57 evolution tests green
  at signoff commit `ddd3fcb`.

---

## B. Code changes (this session, before any credits are spent)

| # | Change | Files (primary) | Status |
|---|---|---|---|
| B1 | Bedrock + base-URL provider wiring, per-role provider flags | `quant_fund_agent/llm.py`, `run_factor_evolution.py` | **DONE** |
| B2 | Token/cost meter + ceiling + persistence + provenance stamps + prompt transcript (`QF_LLM_TRANSCRIPT_PATH`) | `llm.py`, `evolution/loop.py`, `run_factor_evolution.py` | **DONE** |
| B3 | `--config` flag; `--n-tickers 0` = full universe | `run_factor_evolution.py` | **DONE** |
| B4 | Archive cap (`--archive-cap` → `ControllerConfig.archive_cap_per_group`) + eviction lineage events (`controller.event_sink`) | `evolution/controller.py` | **DONE** |
| B5 | Rescore fit-cache (base-fit sharing; byte-identical outputs; full 2A→A+1 impossible without changing numbers — column-order sensitivity) + reveal_index/frontier stamps + rescore diagnostics + `rescore_failed` rows + lineage-resume fix + prequential dedup | `research_eval/harness.py`, `mcp/research_service.py`, `evolution/loop.py` | **DONE** |
| B6 | Fail-closed curation/publish (raises; `QF_PERSIST_FAIL_OPEN=1` escape stamps `curation_failed`/`publish_failed`) | `evolution/loop.py` | **DONE** |
| B7 | Paper harvest query blocks (fundamental/general/price) + `data_scope` tag + `allowed_scopes` retrieval mask + RAG paper cap (`QF_RAG_PAPER_MAX_CHARS`, default 20k chars) | `scripts/populate_papers.py`, `knowledge/embed_store.py`, `knowledge/retrieval.py` | **DONE** |
| B8 | `--creative-frac` (operator `llm_semantic_creative`) + mechanism-brief relaxation (2–4 sentences, non-economic mechanisms OK) | `evolution/loop.py`, `evolution/mutation.py`, `prompts.py` | **DONE** |
| B9 | Per-arm memory keying (`--memory-key`) | `evolution/loop.py`, `run_factor_evolution.py` | **DONE** |
| B10 | `gen_quality.jsonl` per-generation book quality | `evolution/loop.py` | **DONE** |
| B11 | Run configs (panel verified: 3,665 bars × 209 tickers, density 0.463, ends 2024-07-26) | `quant.config.nasdaq100_2010.yaml`, `quant.config.sp500_2010.yaml` | **DONE** |
| B12 | Ablation orchestrator + matrix plan (preflight: forward-reserve, mask density, graph presence, 1-call provider probes) | `run_ablation_matrix.py`, `matrix/final_matrix.yaml` | **DONE** |
| B13 | Walkthrough notebook scaffold (28 cells; renders transcripts, gen_quality, prequential, drift, cost report) | `notebooks/final_run_walkthrough.ipynb` | **DONE** (scaffold; execute against the D3 smoke run) |
| B14 | Fixed statistically-broken curation test (noise unscaled vs 1% returns) | `tests/test_research_eval_curation.py` | **DONE** |

All changes default-off / byte-identical unless flagged, so existing tests stay green.

## C. Run matrix & budget (estimates)

Per-run shape (evolution arms): `--generations 20 --mechanism-groups 4 --demes-per-group 3
--children-per-deme 4` → 960 children + ~50 seed calls; `--population 16`; archive cap 25/group;
`--progressive-reveal --reveal-every 2 --test-frac 0.2`; `--curation greedy --selection-deflation on`;
horizon 6. (~10M in / 1M out tokens debate-on; ~55% of that debate-off.)

| Stage | Runs | Est. LLM cost | Credit pool |
|---|---|---|---|
| Knowledge build (embed + graph) | once | ~$15 | OpenAI |
| Ladder L0–L7 × 2 seeds (Opus 5) | 14 LLM runs + 2 GP + 2 oneshot | ~$700–900 | AWS |
| Model sweep L5 × 5 models × 2 seeds | 10 | ~$590 (Sol 160, Luna 32, Sonnet 60, Fable 300, Muse 34) | OpenAI/AWS/Meta |
| Walk-forward (3 folds, winner) | ~3 run-equivalents | ~$120–225 | winner's pool |
| S&P 500 rerun (winner × 2 seeds) | 2 | ~$60–300 | winner's pool |
| **Total** | | **~$1.5–2.0k** | well within $2.5k OpenAI + $10k AWS + $1k Meta + $0.5k Anthropic |

Wall-clock: bounded by evaluation, not LLMs. Nasdaq-100 PIT 2010–2024.5 ≈ ~3.6k bars × ~200
tickers ≈ 18× the measured 15-ticker panel → estimate ~4–8s/candidate before speedups; **the D2
timing probe must confirm before launch.** Target ≤ ~4h/run, 2–3 runs in parallel.

## D. Verification gates (must pass, in order)

1. **D1 Provider probes**: 1-call smoke per provider (OpenAI Sol/Luna, Bedrock Opus/Sonnet/Fable,
   Meta Muse Spark) through `make_chat_llm`; assert token usage lands in the meter. Verifies keys,
   Bedrock model access, Muse Spark OpenAI-compatibility.
2. **D2 Timing probe**: `run_evolution_timing.py --config-file quant.config.nasdaq100_2010.yaml`
   (2 generations, small children count) → per-candidate eval cost, panel-load time, projected
   run wall-clock, projected $ per model. Abort/rescale if >2× estimates.
3. **D3 Pipeline smoke**: ✅ **DONE in-session (2026-07-27)** — prerun `SMOKE_INTEG` under
   `fmp_archive_equity_nasdaq100pit` (gpt-4o-mini, `--retrieval none`, ridge marginal, 2 gens,
   reveal-every 1, creative-frac 0.5, cap 10, greedy + deflation on, $0.0115): 4 factors
   persisted; verified live — transcripts (10 calls, 4 roles), manifest `llm_roles`/`llm_usage`,
   lineage operators incl. `llm_semantic_creative` + an `archive_evict` event, `gen_quality`
   gens 0–2, prequential `reveal_index` 0–1 (OOS IC 0.061/0.050), factor metadata provenance.
   **Repeat once with graphrag + debate after the graph is built** (retrieval masks + debate
   transcripts are the only paths not yet exercised end-to-end).
4. **D4 Determinism/resume**: kill the smoke run mid-way, resume, assert schedule guard holds and
   lineage/prequential logs are intact (no truncation, no dupes).
5. **D5 Full test suite green** + `test_entrypoint_config_kwargs.py`.
6. Membership-mask density assert (~50% expected) inside the orchestrator preflight.

## D1b. Findings from the in-session integration smoke (2026-07-27, gpt-4o-mini, real panel)

- Whole chain verified live: seeding → in-memory compile → 4-axis evaluation on the real
  Nasdaq PIT panel → coverage gate correctly rejecting a 41%-coverage candidate (with a
  `reasons` dict) → prequential row with `reveal_index`/`frontier_ts` stamps → rescore →
  `gen_quality.jsonl` → per-role cost meter (`llm_usage.json`, $0.006 for the partial run).
- **Evaluation speed evidence (feeds D2)**: with `--marginal-model gradient_boosting` on the
  full 209-ticker panel, one combined-model fit takes on the order of **1–3 minutes** (a 10-min
  run completed only seeding + one reveal). At ~2 fits/candidate × ~1,000 candidates + rescores
  that is **days per run — unacceptable**. D2 must choose the mitigation BEFORE the matrix:
  row-subsampled GB fits, lighter GB params (`fast_model_params` seam exists but is not enabled
  at the research seam), or `--marginal-model ridge` (~seconds/fit; changes the scored axis, so
  whatever is chosen must be identical across ALL arms). The conditioning-factor rationale for a
  nonlinear combiner (CLAUDE.md) argues for subsampled/lighter GB over ridge if affordable.
- Coverage gate vs PIT mask: τ=0.5 is fine (a passing factor shows the denominator is
  mask-aware), but D2/D3 should log the gate pass-rate; if most candidates fail coverage on the
  PIT panel, recalibrate τ rather than letting the search starve.
- Pre-existing, unrelated test failures at clean HEAD (not introduced today, not blocking):
  `test_data_layer.py::test_routed_load_panel_matches_legacy` (LOBSTER synth-field drift) and
  `test_data_layer.py::test_settings_env_override` (QF_DATA_PROVIDER env override not applied).
  Worth fixing in a maintenance pass.

## D2. Next-session runbook (exact order)

```bash
# 0. keys in env/.env: OPENAI_API_KEY, AWS creds (Bedrock model access enabled in
#    the AWS console for Opus 5/Sonnet 5/Fable 5!), META_API_KEY + META_API_BASE_URL.

# 1. Harvest the corpus expansion (+~1000 papers, gpt-4o-mini descriptions):
./venv/bin/python scripts/populate_papers.py --blocks fundamental,general --max-papers 1200
# 2. Re-embed with the OpenAI embedder (hash embedder is the current, weak, cache):
QF_EMBEDDER=openai ./venv/bin/python -c "from quant_fund_agent.knowledge.embed_store import EmbedStore; EmbedStore.build()"
#    (check the exact build entrypoint signature; ~$5, minutes)
# 3. Build the knowledge graph (DOES NOT EXIST YET; graphrag arms crash without it):
./venv/bin/python scripts/build_knowledge_graph.py --model gpt-4o-mini
# 4. Preflight (no credits spent beyond 6 probe calls):
./venv/bin/python run_ablation_matrix.py --plan matrix/final_matrix.yaml --preflight-only
# 5. Timing probe on the real panel (sets/confirms wall-clock + $ projections):
QF_USE_MCP=0 ./venv/bin/python run_evolution_timing.py --config-file quant.config.nasdaq100_2010.yaml \
    --n-tickers 0 --generations 2 --children-per-gen 4
# 6. Smoke run (D3) + kill/resume check (D4), then execute the notebook against it:
QF_USE_MCP=0 QF_LLM_TRANSCRIPT_PATH=... ./venv/bin/python run_factor_evolution.py \
    --config quant.config.nasdaq100_2010.yaml --name SMOKE --model gpt-4o-mini \
    --retrieval graphrag --mechanism-groups 2 --demes-per-group 1 --children-per-deme 2 \
    --generations 3 --progressive-reveal --reveal-every 1 --archive-cap 25 \
    --curation greedy --selection-deflation on --creative-frac 0.2 --max-cost-usd 2
# 7. The matrix (resumable; per-run logs under data/comparisons/final_matrix/logs):
./venv/bin/python run_ablation_matrix.py --plan matrix/final_matrix.yaml
# 8. Walk-forward + S&P 500 rerun for the winner; aggregate; execute the notebook
#    against a real ladder arm for the thesis appendix.
```

Notes for the executing agent:
- Bedrock model IDs (`anthropic.claude-opus-5` etc.) must be *enabled for the AWS account*
  in the Bedrock console first; the preflight probe catches this.
- Muse Spark: verify the Meta Model API is OpenAI-compatible; if not, wire a dedicated
  LangChain integration in `llm.py` (the base-URL hatch assumes OpenAI-compat).
- Bedrock/vendor prices may differ from the table in A1 — update `QF_LLM_PRICES` env
  (JSON: {"model-substring": [in_per_MTok, out_per_MTok]}) so the meter is honest.
- Sonnet-5 intro pricing ends 2026-08-31 — schedule the Sonnet arms before then if possible.
- `--memory` stays OFF except L7 (which uses `--memory-key L7_opus5`).
- If the timing probe projects >8h per run: reduce to `--reveal-every 3`, or cap
  `--n-tickers` with a seeded random subset (needs a small universe.py change), or switch
  `--marginal-model ridge` (changes the scored axis — then it must change for ALL arms).

## E. Deliverables

- `data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/<arm>__s<seed>/` per run
  (factors, evolution state, lineage, prequential, gen_quality, cost report, manifest).
- Aggregation: per-arm and per-model tables (archive quality, prequential OOS IC curves,
  drift across reveals vs generations, PBO, cost per accepted factor) — orchestrator emits
  `data/comparisons/final_matrix/` summary.
- **Walkthrough notebook** (`notebooks/final_run_walkthrough.ipynb`): every stage of one run
  documented — config → panel/PIT mask → seeding (exact prompts in/out) → debate transcript →
  codegen → fitness (all 4 axes + gates + diagnostics) → archive state per generation →
  reveal/rescore event → curation/deflation arithmetic → persisted book → cost report.
  Cells show the actual inputs/outputs of each LLM call (from lineage + stored transcripts).
