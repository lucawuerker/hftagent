# GP factor-mining benchmark (non-LLM baseline)

A deterministic **genetic-programming** alpha miner that serves as a no-LLM
benchmark for the evolutionary LLM factor researcher. In the spirit of
**AutoAlpha** (Zhang et al., IJCAI 2020 — hierarchical evolutionary mining of
formulaic alphas), with **AlphaGen**'s (KDD 2023) *"score a combined set of
factors"* idea folded in for free. It is a benchmark, not new research — the aim
is a *reasonable* baseline the thesis can compare against.

## Why it is a fair benchmark

The evolution machinery is entirely LLM-agnostic — the NSGA-II
`EvolutionController` (selection, Pareto archive, N_trials/deflation, islands,
lineage, checkpoint), the `evaluate_fitness` scoring seam, and `persist_archive`
all consume/produce `FactorProgram` + `FitnessResult` and never touch LLM output.
The GP arm **reuses all of it verbatim** and swaps only the *operator layer*
(seed / mutate / recombine). So the LLM arm and the GP arm differ in exactly one
thing — *how children are proposed* — over the same grammar, data, splits,
fitness vector, selection rule, and persistence. That isolates the contribution
of the LLM proposal mechanism.

Two consequences worth noting for the results chapter:

- **Synergy for free.** In SINGLE mode each candidate's primary axis is its
  **LOCO marginal contribution to the evolving Pareto archive** — i.e. its
  synergistic value to the *collection*, which is exactly AlphaGen's combined-set
  reward, delivered with no extra code.
- **Honest multiple-testing.** GP evaluates far more candidates per generation
  than the LLM arm; every scored candidate bills `next_trial()`, so the deflation
  haircut grows with search effort — an honest exposure of brute-force
  multiple-testing that the shared harness already models.

## The base grammar (and why the GP is confined to it)

The GP mines **typed expression trees** over the project's *base grammar*:

- **terminals** — in-scope data fields (`data["close"]`, …) plus the `returns` /
  `vwap` helper series;
- **windows** — positive-int lookbacks from a fixed pool (for time-series ops);
- **consts** — small floats (for `signed_power` / `power` exponents);
- **operators** — drawn **only** from `factors.ops.BASE_OPS` (an explicit tag in
  `factors/ops.py`) plus the four universal arithmetic operators (`+ - * /`).

Building the grammar from the explicit `BASE_OPS` allowlist — **never**
`dir(ops)` — is what keeps the GP grammar-bound. This matters for the thesis
argument: an **LLM** factor researcher is *not* so confined — a generated factor
may define its own helper functions inline and call arbitrary
`numpy`/`pandas`/`scipy`/`statsmodels`/`sklearn` primitives (see
`codegen.validate_code`'s import allow-list), i.e. it can **extend** the operator
vocabulary on demand. That grammar-extension freedom is a deliberate advantage of
the agentic framework and must remain exclusive to the LLM arm. New ops are
**opt-in** to `BASE_OPS`; anything the LLM/framework adds later is off-limits to
the GP by construction, and `tests/test_gp_grammar.py` asserts
`used_ops(tree) ⊆ ops.BASE_OPS`.

`indneutralize` (needs a `sector` field / cross-sectional GROUP type) is excluded
from the v1 grammar for robustness; it is a documented extension.

## Hierarchical growth (AutoAlpha)

`--depth-schedule 3,5,7` runs the search in stages of increasing tree-depth cap,
carrying the archive across stages so deeper factors recombine shallow winners —
AutoAlpha's short→long hierarchical search. Each stage runs `--generations`
generations. AutoAlpha's PCA-directed novelty seeding is replaced by the harness's
residual-IC / independence axis, which already rewards decorrelated content.

## Files

| File | Role |
|---|---|
| `factors/ops.py::BASE_OPS` | the base-grammar tag (the confinement boundary) |
| `agents/factor_research/gp/grammar.py` | typed nodes, operator table, `random_tree` (ramped half-and-half) |
| `agents/factor_research/gp/render.py` | tree → validator-passing `BaseFactor` module / `FactorProgram` |
| `agents/factor_research/gp/operators.py` | subtree crossover, subtree/point/hoist mutation (typed, depth-capped) |
| `agents/factor_research/gp/loop.py` | `GPRunConfig` + `GPLoop` (reuses controller + eval seam + checkpoint) |
| `run_gp_factor_mining.py` | entrypoint (mirrors `run_factor_evolution.py`) |
| `evolution/loop.py::persist_archive` | `engine`/`model_label` kwargs → honest `engine="gp"` provenance |

## Running it

```bash
# Quick, in-process (no LLM, no server). Writes a normal prerun.
python run_gp_factor_mining.py --name gp1 --config-name yfinance_equity_sp100 \
    --generations 3 --depth-schedule 3,5 --seed-pop 40 \
    --children-per-generation 16 --n-tickers 15

# Driven by a data config file (sets QF_CONFIG_FILE before the panel loads).
python run_gp_factor_mining.py --name gp-sp100 --config quant.config.sp100.yaml \
    --generations 6 --depth-schedule 3,5,7 --seed-pop 60

# Compare the GP prerun head-to-head with any LLM prerun.
python run_model_comparison.py --preruns gp-sp100,<llm-prerun> --fast
```

The run persists its Pareto archive as a standard prerun under
`data/workspaces/<config>/preruns/<name>/factors/factor_db.json` (records tagged
`source=researcher`, `engine=gp`, with the same `metadata.evolution` provenance
the LLM arm writes) plus GP state/lineage under `<scope>/gp/`. The main seed DB is
never written. The comparison and rolling harnesses ingest it exactly like any
oneshot/evolution prerun, so the GP arm slots into the ablation matrix as the
non-LLM baseline row.

## Optional extensions (documented, not built in v1)

- `--unit set`: SET-mode collections scored jointly via `evaluate_set_fitness`,
  with structural add/drop/replace GP ops (closest to AlphaGen's explicit set
  reward).
- A canonical single-objective AutoAlpha fitness (IC + max-|corr| decorrelation)
  as an additional literature-standard baseline.
- `indneutralize` / cross-sectional GROUP terminals when `sector` is in scope.
- PCA-directed seeding (AutoAlpha's principal-alpha initialisation).
