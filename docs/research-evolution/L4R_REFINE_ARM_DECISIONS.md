# L4R — non-evolutionary "refine" arm: decision record (2026-07-31, autonomous)

User request (verbatim intent): test the suspicion that the evolutionary
machinery adds little over a good model, by running an arm against the finished
`L4_terra_s0` (GPT-5.6 Terra, 44 factors, 798 trials, $86.70, 903 min) that
drops evolution but keeps the knowledge graph with broad coverage, ALL
statistical safeguards incl. progressive reveal, 4-axis Pareto scoring, LLM
refinement of the SAME factor based on its score, and occasional combination of
factors from different knowledge-graph areas.

## What was built

`--variant refine` on `run_factor_evolution.py` (loop-level, default `evolve`
is byte-identical). Refine mode replaces the evolutionary operators only:

- Every seeded factor starts a **lineage**; each lineage is refined by the LLM
  against its own deterministic evaluation report at most `--refine-rounds`
  times (new prompt `build_refine_prompt`: *same factor, same mechanism, better
  implementation* — explicitly forbidden from switching mechanism, unlike the
  mutation prompt).
- A deme that runs out of refinement work **re-seeds fresh graphrag-grounded
  ideas** from the paper corpus every generation (broad knowledge-graph
  coverage keeps growing instead of iterating deeper).
- The only combination operator is the occasional **cross-group synthesis**
  (`--p-cross-group 0.10`, parents picked by the existing Pareto tournament);
  no same-group crossover, no jitter mutation, no migration, no tournament-
  driven descent.
- Unchanged: harness scoring, 4-axis Pareto vector, gates, per-group archives +
  cap, progressive reveal (prequential probe, archive rescore, gate-failer
  retry), N_trials billing, curation, publish-time deflation, persist path.
- Resume-safe via `evolution/refine_state.json` (lineage → refines used),
  checkpointed every generation next to `state.json`.

Tests: `tests/test_evolution_refine.py` (5 tests); plan-vs-argparse guard now
also covers `matrix/terra_l4.yaml` + `matrix/terra_l4_refine.yaml`.

## Decisions taken without you (review these)

1. **D1 — same generations/reveal schedule as L4 (20 gens, reveal-every 2,
   test-frac 0.2, seed-frac 0.45).** Keeps the calendar splits byte-identical
   to `L4_terra_s0`, so prequential rows and rescore diagnostics are directly
   comparable generation-by-generation. Refinement work is paced across the
   whole schedule (one refinement per lineage per generation max), so
   refinements react to newly revealed data via the pre-refinement rescored
   brief.
2. **D2 — children-per-deme 1 (vs L4's 2), refine-rounds 2, seeds 12/group.**
   The arm is deliberately CHEAPER (~24 children/gen vs 48): the hypothesis
   under test is that first solutions are good and heavy iteration is wasted
   spend. Expected ~550–650 scored trials vs L4's 798, est. ~$55–70 vs $86.70.
   Deflation is N_trials-aware, so each arm is billed honestly for its own
   number of looks. Comparison at matched-trial counts is still possible from
   `lineage.jsonl`/`gen_quality.jsonl` curves.
3. **D3 — steady-state slot mix instead of "seed everything at gen 0".**
   96 gen-0 seeds; afterwards each deme's one slot per generation goes to
   (a) cross-group synthesis w.p. 0.10, else (b) the deme's least-refined
   pending lineage, else (c) a fresh graphrag seed. So paper extraction runs
   all through the run, not only at the start.
4. **D4 — a failed refinement still consumes a round** (the lineage keeps its
   previous version); duplicates/compile-failures are not retried forever.
5. **D5 — cross-group children start their own lineage** (they may be refined
   up to refine-rounds like any seed), and their parents are chosen by the
   existing NSGA-II tournament *within* each group — i.e. Pareto scores are
   used to pick what to combine, but never to breed within a group.
6. **D6 — refinement of gate-FAILING factors is allowed** (the report tells
   the LLM what failed; that is the point of refinement). Exhausted lineages
   whose versions all fail gates simply die out of the queue.
7. **D7 — `--final-holdout` left OFF** (as in L4_terra_s0, which launched
   before that flag existed) so the two arms' schedules match exactly.
8. **D8 — creative-frac kept at 0.1** (seeding parity with L4: ~10% of seed
   ideas are knowledge-only "creative" rather than paper-grounded).
9. **D9 — budget**: separate plan `matrix/terra_l4_refine.yaml`,
   `budget_usd: 120` hard cap from the OpenAI credit pool (same pool as the
   Terra L4 plan, disjoint accounting). Arm name **`L4R_terra_s0`**, seed 0.
10. **D10 — population/archive knobs unchanged** (population 16/deme, archive
    cap 40/group, curation archive, selection-deflation on, lightgbm marginal
    model, formulaic-101 fixed + reference books, n-tickers 0 = full panel).
11. **D11 — a ~$0.80 live smoke (`refine_smoke` prerun, capped $3)** was run
    before launching the real arm, then deleted.
12. **D12 — generation-0 seeding hardened for the 30-min task reaper** (applies
    to ALL evolution runs, evolve included): seeding now checkpoints after
    every mechanism group, and a checkpoint with `generation == 0` and a
    non-empty population resumes mid-seeding, skipping already-seeded groups
    (`loop._admit_seed_group`, entrypoint resume gate relaxed to
    `ctrl.population()`). Motivated live: the first L4R launch was killed
    ~28 min in with 5/8 groups seeded and the old code would have re-seeded
    from scratch on every relaunch. That first attempt's ~$3–5 of seeding
    spend predates the first usage checkpoint, so it is NOT counted by the
    plan's budget accounting (OpenAI-side it was spent). Semantically the
    restructure is order-identical for uninterrupted runs; guarded by
    `test_mid_seeding_kill_resumes_without_reseeding_done_groups`.

## How to compare (after the run)

Standard post-analysis suite on `L4R_terra_s0` (analyze → figures → combined
book backtest → prequential deployment), then side-by-side vs `L4_terra_s0`:
final book OOS (walk-forward/prequential), combined-book backtest, per-trial
quality curves, cost/tokens, archive size + mechanism coverage.
