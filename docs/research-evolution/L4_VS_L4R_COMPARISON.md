# Evolution vs refinement-only: L4_terra_s0 vs L4R_terra_s0 (2026-08-01)

Question (user, 2026-07-31): does the evolutionary machinery add enough value
over "strong model + knowledge graph + statistical harness + light same-factor
refinement" to justify its cost?

Both arms: GPT-5.6 Terra, seed 0, Nasdaq-100 PIT 2010→2024-07 (2-year forward
reserve), graphrag + 8 mechanism groups (max mode), 20 generations, identical
progressive-reveal schedule (reveal-every 2, test-frac 0.2), archive curation +
publish-time deflation, formulaic-101 fixed/reference book, lightgbm marginal
model. Only the operator layer differs (see
`L4R_REFINE_ARM_DECISIONS.md`); L4R was deliberately given a smaller child
budget (1 child/deme/gen vs 2).

## Cost & scale

|                        | L4 (evolve) | L4R (refine) | ratio |
|------------------------|------------:|-------------:|------:|
| scored trials          | 798         | 468          | 0.59  |
| LLM cost               | $86.70      | $59.71 (*)   | 0.69  |
| tokens in / out        | 13.9M / 3.5M| 8.8M / 2.5M  | 0.65  |
| wall clock             | 903 min     | 329 min      | 0.36  |
| final book (persisted) | 44          | 62           |       |
| groups covered         | 8/8         | 8/8          |       |

(*) plus ~$3–5 unmetered from the first killed launch (pre-checkpoint; D12).

Operator mix — L4: 354 llm_semantic + 188 crossover + 74 cross_group +
73 jitter + 33 creative (archive origin dominated by mutation chains, median
origin generation 14). L4R: 263 refine + 84 mid-run fresh seeds + 42
cross_group (archive origin 43 refine / 14 seed / 5 cross_group, median origin
generation 11.5). Refinement helped modestly and reliably: 54% of the 263
refinements improved the lineage's marginal value (median delta +0.001), and
every refinement child passed the gates.

## In-search metrics (data selection touched — inflated for both)

Final dev-window archive: L4 mean marginal 0.0019 / max 0.0357, combined
lightgbm VAL IC 0.104. L4R mean marginal 0.0002 / max 0.0125, combined VAL IC
0.077. **Evolution clearly wins the metric it optimises.** In-panel net Sharpe
of the combined book: L4 2.68 vs L4R 1.15 — read together with the forward
numbers below, most of that gap is in-panel overfitting, not real edge.

## Honest out-of-sample

**In-run prequential probes** (each reveal block scored before selection ever
saw it; identical calendar blocks): mean combined OOS IC L4 **+0.060** vs L4R
**+0.064**; late-run PBO lower for L4R (last three blocks 0.11/0.14/0.21 vs
0.23/0.50/0.31). At 59% of the trials the refine arm's honest in-run OOS track
is at least as good.

**Combined-book backtest, 2-year forward reserve (net, model race on VAL):**

| construction    | metric        | L4        | L4R       |
|-----------------|---------------|----------:|----------:|
| cross-sectional | forward IC    | +0.0396   | +0.0303   |
|                 | forward Sharpe| 0.469     | **0.661** |
|                 | forward ann.  | 3.3%      | **5.9%**  |
|                 | DSR prob      | 0.503     | **0.773** |
|                 | race PBO(VAL) | 0.143     | 0.837     |
|                 | winner        | rand.for. | ridge     |
| per-underlying  | forward Sharpe| **0.254** | −0.526    |
|                 | DSR prob      | 0.559     | 0.145     |

**Prequential deployment replay** (book as of each generation traded on the
next unseen block; combiner fixed to lightgbm a priori; identical TEST and
forward segments):

| segment (cross-sectional)  | L4      | L4R      |
|----------------------------|--------:|---------:|
| stitched reveal blocks     | −0.19   | **+0.22**|
| never-revealed TEST tail   | −0.18   | **+0.40**|
| 2-y forward reserve        | **+0.70**| +0.50   |
| per-underlying TEST        | +0.03   | −0.29    |
| per-underlying forward     | +0.36   | +0.50    |

(Sharpe, net. The stitched-blocks segments cover different bar counts because
the two books enter deployment at different points; TEST/forward are
identical windows.)

## Reading

- The user's suspicion is **substantially supported for the deployed
  (cross-sectional) construction**: at 69% of the cost, 59% of the statistical
  trials and a third of the wall clock, the refine-only arm produced a broader
  book (62 factors, all 8 mechanism groups) whose honest OOS record —
  prequential blocks, never-revealed TEST tail, forward-reserve Sharpe and
  deflated-Sharpe probability — is as good as or better than the evolutionary
  arm's, with a far smaller in-panel→forward overfitting gap (1.15→0.66 vs
  2.68→0.47).
- Evolution is not worthless: it wins every *in-search* metric (deeper
  marginal-value optimisation), the stitched cross-sectional forward segment
  (0.70 vs 0.50), and the per-underlying construction, where the refine book
  is clearly worse. If the per-underlying deployment mattered, evolution's
  extra depth pays; for this universe the position-construction default is
  cross-sectional.
- Caveats: single seed, single model (Terra), and L4R's child budget was
  intentionally smaller — this is a cost-effectiveness comparison, not a
  matched-compute operator ablation. The high race-PBO (0.84) behind L4R's
  winning ridge combiner means the forward Sharpe edge should be quoted with
  the DSR probability (0.77), not alone.

Artifacts: `data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/L4R_terra_s0/`
(analysis_report.md, 22 figures, book_backtest/, prequential_deployment/) —
same layout as L4_terra_s0. Matrix row appended to
`data/comparisons/final_matrix/matrix_summary.csv`.
