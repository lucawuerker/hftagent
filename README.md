# QuantFundAgent

Code accompanying the Imperial College London MSc thesis **"Literature-Grounded
Language-Model Agents for Alpha Factor Research"** (Mathematics & Finance,
2025–2026).

The framework lets language-model agents perform alpha factor research the way
a human team would — read the literature, form economic hypotheses, implement
them as code, and learn from evaluation feedback — while a fixed, deterministic
harness decides what survives. The agents never grade their own proposals:
every candidate factor is scored on a four-objective Pareto vector
(marginal value, independence, parsimony, structural novelty), every evaluated
candidate is counted toward a multiple-testing correction, and the final factor
collections are measured on a five-year walk-forward record in which each
half-year block is scored *before* any part of the system may adapt to it.

## How it works

1. **Corpus & knowledge graph** — an arXiv harvester builds a corpus of
   research papers; an extraction pass turns them into a
   Paper → Mechanism → Factor → Field graph (`quant_fund_agent/knowledge/`).
   Community detection and gap queries reveal which documented economic
   mechanisms no existing factor exploits yet.
2. **Grounded ideation** — mechanism communities steer hypothesis generation;
   each hypothesis cites the retrieved papers it rests on and can pass an
   adversarial review before implementation
   (`quant_fund_agent/agents/factor_research/`).
3. **Code generation & validation** — hypotheses become `BaseFactor` Python
   programs, validated in memory (schema, field scope, causality gate that
   recomputes each signal on a truncated panel to refuse look-ahead).
4. **Deterministic evaluation** — `quant_fund_agent/research_eval/` scores
   every candidate: LOCO marginal contribution to the combined forecast of the
   evolving collection, residual-IC independence, AST-based parsimony and
   structural novelty, plus deflated statistics (deflated IC / Sharpe, CSCV
   probability of backtest overfitting) fed by an explicit trial count.
5. **Evolution** — a constrained NSGA-II controller with knowledge-graph
   mechanism groups and demes evolves the population; LLM mutation/crossover
   prompts receive a deterministic reflection brief built from the candidate's
   own diagnostics (`quant_fund_agent/agents/factor_research/evolution/`).
6. **Progressive reveal / walk-forward** — the data window is revealed
   block-by-block across generations; each newly revealed block is scored
   prequentially before the archive may adapt to it, producing an honest
   out-of-sample record per run.
7. **Baselines** — a genetic-programming factor miner over a fixed typed
   grammar (`agents/factor_research/gp/`) and the 101 published formulaic
   alphas (`data/prebooks/formulaic_101.json`) serve as the non-LLM and
   published-benchmark comparisons, evaluated by the identical harness.

## Repository layout

| Path | Contents |
| --- | --- |
| `quant_fund_agent/` | The framework package: data layer, knowledge graph, researcher agents, evolution loop, deterministic evaluation harness, GP baseline |
| `run_factor_evolution.py` | Main entrypoint: one evolutionary (or refine-variant) research run |
| `run_factor_research.py` | One-shot researcher (no evolution) |
| `run_gp_factor_mining.py` | Genetic-programming baseline |
| `run_ablation_matrix.py` + `matrix/` | Orchestrator and run plans of the thesis ablation ladder |
| `run_model_comparison.py` | Factor-set comparison harness (IC, analytics, combined-signal backtests) |
| `effective_factors/` | The selected effective factors from the thesis runs (per book and union, full implementations + selection summary) |
| `data/knowledge/` | Knowledge graph, paper embeddings, and the frozen graph snapshots the thesis runs used |
| `data/factors/`, `data/prebooks/` | Seed factor library and the 101-formulaic-alphas benchmark book |
| `notebooks/evolution_walkthrough.ipynb` | Guided, re-runnable walkthrough of one research generation |
| `scripts/` | Corpus harvesting, graph building, run analysis |
| `tests/` | Test suite (`./venv/bin/pytest`) |

## Getting started

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # or create .env with OPENAI_API_KEY=...
```

The fastest way to see the whole loop is the walkthrough notebook
`notebooks/evolution_walkthrough.ipynb`: it runs from a fresh clone (knowledge
graph, paper index and embeddings are committed; market data comes from the
free yfinance path) and needs only an OpenAI key in `.env` — a full pass costs
roughly $1–2 and is capped by a hard cost ceiling.

A small live run:

```bash
./venv/bin/python run_factor_evolution.py --name demo --retrieval graphrag \
  --mechanism-groups 2 --demes-per-group 2 --generations 3 --max-cost-usd 5
```

Universe, date range and data provider are configured through
`quant.config.*.yaml` (see `python -m quant_fund_agent.setup` for the wizard).

## Effective factors

`effective_factors/` contains the factor implementations selected in the
thesis: the per-arm books, the benchmark, and the QR-pivoting selection of the
union book, together with `selection_summary.json` describing how each subset
was chosen. Each factor is a standalone `BaseFactor` subclass and can be
evaluated on any configured panel.

## Thesis experiments

The ablation ladder (grounding × evolution × review, against the GP baseline
and the 101 formulaic alphas) is specified by the plans in `matrix/` and driven
by `run_ablation_matrix.py`. The analysis scripts that produced the thesis
tables and figures live in `scripts/` (`wf_arm_factor_analysis.py`,
`wf_pit_combiner_study.py`, and related). Result artifacts are not tracked in
this repository.
