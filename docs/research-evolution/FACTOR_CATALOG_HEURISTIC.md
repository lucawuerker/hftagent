# Factor-catalog selection heuristic

**Status: agreed with Luca 2026-07-29.** Turns the union of all runs' kept-pools
(every gate-passer ever scored) into the product catalog that request-time
strategy construction selects from. Generous by design: better too many than
too few — only true copies die. Implemented in
`scripts/build_factor_catalog.py`.

## Principles

1. **AST similarity is the primary dedup** (structural clones are the one
   thing evolution mass-produces). Behavioral (correlation) pruning is
   secondary and applied **only within a mechanism/category bucket** — two
   correlated factors from *different* mechanisms both stay.
2. **Depth over parsimony**: every populated bucket keeps multiple genuinely
   different expressions; there is no global size cap.
3. **Zero new trials**: the catalog is built from *already-persisted*
   diagnostics only (no new model fits), and it carries the global
   `n_trials` of its contributing runs as metadata — every downstream
   strategy gate must deflate against it.

## Stages

- **Stage 0 — eligibility.** Computable on the current field scope; recorded
  coverage ≥ 0.5; a finite recorded objective. Pure filters.
- **Stage 1 — structural dedup (PRIMARY).** Canonical-AST similarity
  (`research_eval.ast_novelty`, numeric-literals normalised so window tweaks
  are clones). Factors with pairwise AST similarity ≥ `--ast-sim` (default
  0.95) form a clone family; keep the best-scoring member per family.
- **Stage 2 — within-bucket behavioral dedup.** Bucket = category, refined by
  the knowledge-graph mechanism tag when present. Inside each bucket, greedy
  by score: a candidate is dropped only when **two** already-kept factors of
  the same bucket correlate with it at |ρ| ≥ `--corr` (default 0.85) on the
  panel — i.e. every high-correlation cluster keeps its top TWO members
  ("keep two per cluster, only kill after that"). No cross-bucket pruning.
- **Score (rank-only).** Recorded marginal value at evaluation, tie-broken by
  coverage and |degradation ratio| sanity. Sign-agnostic (a factor's
  `expected_sign` is respected downstream; conditioning factors stay).

## Output

`data/books/catalog_<name>/catalog.json`: one row per surviving factor with
id, code provenance (run/arm, generation, operator), category, mechanism,
bucket, clone-family id, correlation-cluster neighbours, score components,
and the catalog-level `n_trials_global`. Per-universe catalogs (signals do
not transfer across universes).
