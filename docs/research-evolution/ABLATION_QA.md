# LLM-contribution ablation (2026-08-09, COMPLETE — final table at the end)

**Questions** (user, 2026-08-09):
1. What does the LLM contribute **inside the evolution** (mutation/crossover),
   beyond generation-0 ideation? Hypothesis: ideation + deterministic
   machinery suffices.
2. How much does the **evolutionary loop** add over one-shotting — or is the
   deterministic scoring harness the real source of the edge?

**Design** (aligned with user): decompose along
`ideation → +deterministic scoring → +deterministic evolution → +LLM
evolution`, all as FULL walk-forward ladder arms (identical two-phase
schedule as `matrix/terra_wf_ladder.yaml`: 20 generations, `--wf-blocks 10
--wf-block-bars 126`, graph-readonly snapshot, seeding parity 12 ideas × 8
groups = 96 except L1HB). Fairness by **reported N_trials deflation**, not
trial matching (only the GP arm is trial-matched for free). Plan:
`matrix/ablation_qa.yaml` (guarded by `tests/test_final_matrix_plan.py`).

| Arm | Isolates | Mechanics | Status |
|---|---|---|---|
| L1WF (existing) | ideation only | oneshot, no harness in the loop | done (ladder) |
| **L1H_terra_s0** | + deterministic scoring | `children-per-deme 0` (new `max(0,…)` seam in `evolution/loop.py`): gen-0 graphrag seeding, then ONLY prequential/rescore/prune/curation | **done 2026-08-09 (local M2)** |
| **L4D_terra_s0** | + deterministic evolution | children ONLY from AST window-jitter: `--p-llm 0 --p-crossover 0 --p-cross-group 0 --p-jitter 1` (pure flags, no code) | running (local M2) |
| **L0WF_gp_s0** | no LLM anywhere | GP miner with progressive reveal **ported into `gp/loop.py`** (2 depth stages × 10 gens = 20 windows; seed_pop 96, 36 children/gen ≈ trial-matched to L4WF) | running (server) |
| L4WF (existing) | + LLM evolution | full loop | done (ladder) |
| **L1HB_terra_s0** | archive-size scaling | L1H with `--seed-ideas-per-group 24` (=192 ideas) → target ≥38-factor final archive; answers "what does a 40–50-factor oneshot book cost?" | queued (local, after L1H PIT race) |

## Results so far

**L1H vs L4WF — the headline (single seed, caveats below):**

| | L1H (ideation+scoring) | L4WF (full LLM evolution) |
|---|---|---|
| prequential mean IC (10 honest WF blocks 2021-26) | **+0.0317** | +0.0352 |
| hit rate | 80 % | 80 % |
| trials billed (deflation input) | **81** | ~800 |
| LLM cost | **$9.53** | ~$250 |
| final archive | 23 | 57 |
| funnel | 96 ideas → 81 scored → 50 admitted → 23 survive 20 rescores | — |

Reading: at 4 % of the cost and 1/10 the trials, ideation + deterministic
scoring is statistically indistinguishable from full evolution on the honest
walk-forward record (Δ=0.0035, per-arm block SE ≈ 0.012). Deflation favors
L1H further. Consistent with two prior findings: the unfiltered L1WF book
already rivals L4WF under the PIT combiner race (lasso 0.078 vs 0.072), and
L4R (refine, no breeding) matched evolution at 69 % cost.

Caveats: one seed; prequential combiner is the runs' lightgbm (PIT race with
lasso/ic pending → `wf_arm_analysis_local/`); L4WF's snapshots grow through
the walk-forward while L1H's book is fixed after gen 0 (creation-time PIT is
honest for L1H — every factor predates every reveal — but archive-membership
dynamics differ).

**L1H PIT combiner race** (kept book 80 F., `--availability full` — every
factor predates the reveals; local run, `wf_arm_analysis_local/`):
ic **0.0749**, lasso 0.0745, equal 0.0632, ridge 0.0603, rf 0.0543,
autoalpha 0.0408 — all 10/10 blocks positive. Three-way picture under the
best combiners: L1WF raw oneshot 0.078 ≈ **L1H 0.075** ≈ L4WF full evolution
0.072 (differences ≪ block noise) at $50 / $9.5 / $250.

**L0WF (GP) — METRIC GAMING, not alpha (2026-08-09).** The GP arm "won" its
prequential record (+0.144 mean, 100% hit) via `gp_0187 = log(high)` —
the raw price level, per-block per-underlying IC −0.28…−0.37. This is an
integration artifact: the level is the cumulative sum of returns, so its
correlation with future returns telescopes short-horizon mean reversion
over every lag in the window (an honest k-day reversal factor measures the
same effect at ~0.05). Much of the rest of the book is the same class (raw
quarterly fundamental levels). No mechanical look-ahead (code scanned).
**Thesis reading: without a semantic prior, the prior-free operator
maximises fitness by finding degenerate statistics the per-underlying IC
convention rewards for non-stationary signals — the LLM's economic prior
acts as implicit regularisation.** Exposes a harness gap: no stationarity
gate (options: document only / AR(1)-based level gate + GP rerun / both —
user decision pending).

**Level-gate diagnostic (2026-08-09, decision: variant c).** Measured (never
re-run) over every ladder archive: median per-name lag-1 autocorrelation
(``rho_med``) per factor on the dev window; gate threshold **rho > 0.995**
separates the pathology class cleanly. Fails per book: zoo **0**/101,
L1H 1/23, L1WF 4/105 (4 %), L4WF 4/57, L6WF 2/31, L2WF 4/19 — vs **GP
6/14 (43 %**, incl. ``log(high)`` at lvl_corr 0.999**)** and, notably,
L5WF 9/35 + L7WF 11/42 with rho = 1.0 exactly: quarterly-stepped
fundamentals/event signals (not price-integration artifacts; flagged for the
thesis discussion, no reruns). Full table:
``data/comparisons/wf_arm_analysis/level_gate_diagnostic.csv`` +
``scripts/level_gate_diagnostic.py``. Implementation: harness now emits a
``level_rho`` diagnostic for EVERY candidate (all arms, measure-only);
the GP loop gates on it via ``--level-rho-max`` (default off). **L0WFG_gp_s0**
(gated rerun, ``level-rho-max 0.995``) launched on lagias — the ungated
L0WF stands as the metric-gaming evidence, L0WFG is the fair no-LLM baseline.
L4D is marked ``after: RUNS_LOCALLY`` in the plan (executes on the M2).

**L0WFG (gated GP rerun) — the gate was EVADED (2026-08-09).** With
``level-rho-max 0.995`` the gate fired 213 times, yet the arm still posts a
too-good record (prequential mean +0.098, 100 % hit, 814 trials, 8-factor
final book) via near-boundary level proxies — flagship ``gp_0987 =
rolling_residual(grahamNumber, epsDilutedGrowth, 40) − marketCap``: dominated
by −marketCap (a price-level in disguise), with the residual term adding fast
noise that pushes median per-name rho just under the threshold while the
integrated component keeps inflating the per-underlying IC (single-factor
block IC up to 0.31). **Decision: no further gate iterations — this is an
adversarial arms race a simple statistic cannot win, and that IS the
finding: a prior-free search games any imperfect metric (Goodhart);
the LLM's economic prior is load-bearing regularisation, which is what the
GP baseline was supposed to measure.** Both GP records (ungated 0.144,
gated 0.098) are reported as metric-gaming evidence, not as alpha.

## Data inventory (for future analysis sessions)

- **Preruns** `data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/<arm>/`
  (local M2 = source of truth for L1H/L4D/L1HB, mirrored to lagias; L0WF on
  lagias). Evolution arms: `evolution/{state.json, lineage.jsonl,
  prequential.jsonl, gen_quality.jsonl, llm_usage.json, run_config.json}` +
  `factors/factor_db.json`. **GP arm writes to `gp/` not `evolution/`.**
- **PIT combiner race outputs**: server
  `data/comparisons/wf_arm_analysis/pit_combiners/` (15 books × 9 methods ×
  10 blocks, jsonl + `_summary.csv` + `artifacts/<label>/g<gen>/`
  weights/models/OOS-block-predictions + shared `signal_store/`); local runs
  under `data/comparisons/wf_arm_analysis_local/`. L1H raced with
  `--availability full` (see `scripts/wf_pit_combiner_study.py`).
- **Analyses already done** (2026-08-05→08, report artifact on claude.ai
  "PIT Walk-Forward Combiner Race"): 15-book combiner leaderboard (lasso
  0.071 > ic 0.064 > ridge 0.063 > autoalpha 0.057 > rf > lightgbm ≫
  Kakushadze), lasso sparsity/turnover/never-selected analysis, strategy lab
  (`scripts/wf_pit_strategy_lab.py`: frozen per-name construction + index
  hedge → L4WF/lasso net Sharpe 1.19–1.39 at 0.002/day turnover; cs-MN
  construction inverts the ranking, rf 0.92).
- **Logs/supervisors**: local `data/{l1h,l4d,l1hb}_local_run.log` +
  `scripts/{l1h,l4d}_local_supervisor.sh`, `scripts/l1h_pit_then_l1hb.sh`;
  server lanes `data/ablation_qa_lane{A,B,C}.log`.
- **Dashboards**: `http://31.97.141.166:8899/wf-2b505d86a0f2/ablation.html`
  (server arms), `.../l1h_local.html` (local pushes via
  `scripts/l1h_status_push.sh`), `.../analysis.html` (combiner campaign).

## FINAL RESULTS (all arms complete, 2026-08-09; single seed)

Honest prequential record (mean of the 10 walk-forward block ICs 2021-26,
runs' own lightgbm combiner) + PIT-race best where computed:

| Arm | Adds | mean IC | hit | trials | LLM $ | final book |
|---|---|---|---|---|---|---|
| L1WF | ideation only | (no prequential; PIT race lasso **0.078**) | — | 107 | 20.86 | 107 |
| L1H | + det. scoring | +0.0317 (race ic 0.075) | 80 % | 81 | 9.53 | 23 |
| L1HB | + det. scoring, 2× seeds | **+0.0438** | 70 % | 172 | 16.18 | 25 |
| L4D | + det. evolution (jitter) | +0.0285 | 70 % | 621 | 9.62 | 46 |
| L4WF | + LLM evolution | +0.0352 (race ic 0.073) | 80 % | ~800 | ~250 | 57 |
| L0WF / L0WFG | GP, no LLM | 0.144 / 0.098 — **metric-gamed**, not alpha | — | ~813 | 0 | 14 / 8 |

**Answers (this setup: Nasdaq-100 PIT daily, Terra-class model, 1 seed):**
1. *LLM inside the evolution*: no measurable honest-OOS value beyond
   generation-0 ideation. L4D (deterministic jitter children, 621 trials)
   and L1H (no children at all, 81 trials) both land within block noise of
   L4WF (~±0.005 around 0.03, SE≈0.012). The LLM's decisive contribution is
   the **economic prior at ideation** — demonstrated a contrario by the GP,
   which without it games every metric (level artifact, then gate evasion).
2. *Evolution vs one-shot*: the edge attributed to "evolution" is carried by
   **ideation + the deterministic scoring machinery** (gates, rescoring over
   the reveals, curation, deflation). Best honest record of the family:
   L1HB at $16 — more seeding bought a *better* archive, not a bigger one
   (Pareto front saturates at ~25; breadth is a curation lever on the
   kept_pool, not a seeding lever). Trial counts (81 vs 621 vs 800) do not
   correlate with the honest record — the anti-ratchet machinery works.
   Caveats for the write-up: single seed; in-search metrics favour evolution
   (the thesis' overfitting story); books differ in breadth (23–57), which
   matters for downstream combination (union results).

## Pending

- L4D + L0WF completion → same prequential comparison + PIT race per book.
- L1HB → empirical cost of a 40–50-factor selection-only book (projection
  from L1H: ~24 % of ideas survive to the final archive → 192 ideas ≈ $19–20
  all-in, still <10 % of L4WF).
- Seed-1 replications (all arms are single-seed so far); s1 ladder arms
  remain on HOLD in `matrix/terra_wf_ladder.yaml`.
- Server CPU-steal issue: Hostinger fair-use throttle re-engaged 3× (50 %
  steal even at 2-process load) — heavy compute has moved to the local M2;
  reassess hosting for future campaigns.
