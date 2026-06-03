# Pipeline showcase (no market data required)

These scripts let you watch the **creative** stages of the quant-fund agent
pipeline run end-to-end **without the proprietary order-book / stock-price data**
(`ticker_data/`), which is not part of this repository.

Each agent runs *exactly as it does in production*, but stops at the boundary
where it would need the market-data panel. So you get to see real factors and
real strategies being **researched and created** — just not back-tested or
validated.

| Stage | Script | Runs | Stops before (needs data) |
|------|--------|------|---------------------------|
| 1. Factor Researcher | `1_factor_research.py` | reads papers (full text) → brainstorms ideas → writes `BaseFactor` subclasses | the IC back-test |
| 2. Selector | `2_selector.py` | builds a hypothesis → selects factors (reasons over economics, no IC) | — (needs no market data; runs fully) |
| 3. Architect | `3_architect.py` | designs a strategy (model + features + position rules) | the fit + back-test refinement loop |

The **Statistician** and **Portfolio Manager** stages are intentionally not
included here: they exist only to *validate and allocate* using the market-data
back-tests, so there is nothing meaningful for them to do without the data.

## Prerequisites

* The project virtual environment.
* An OpenAI API key — the agents call an LLM to brainstorm, select and design.
  Put it in a `.env` file at the repository root:

  ```
  OPENAI_API_KEY=sk-...
  ```

## Running

From anywhere in the repo (the scripts `cd` to the project root themselves):

```bash
./venv/bin/python showcase_pipeline/1_factor_research.py
./venv/bin/python showcase_pipeline/2_selector.py
./venv/bin/python showcase_pipeline/3_architect.py
```

Run them in order for a coherent story (the factors created in Stage 1 become
selectable in Stage 2, whose hypothesis feeds Stage 3), or run any one on its
own — each is self-contained.

## Reading papers (full text from the whole index)

The Factor Researcher draws from the **entire** paper index
(`data/papers/index.json` — hundreds of papers), not just the handful of PDFs
committed to the repo. For any selected paper that has no local PDF, it fetches
the **full text from the paper's coupled link** (e.g. the arXiv `pdf_url`) rather
than falling back to the abstract. Stage 1 prints each paper's text source
(`local PDF`, `full text fetched via paper link`, or `abstract only`).

Fetched text is cached under `data/papers/fulltext_cache/<id>.txt`, so each paper
is downloaded at most once. This needs internet access; to run fully offline set
`PAPER_FETCH_FULLTEXT=0` (papers without a local PDF then fall back to their
abstract).

## What gets written, and where

The scripts **never touch the real databases**. They read from / write to
*separate showcase copies* so the committed data stays clean:

* **Generated factor subclasses** → `quant_fund_agent/factors/researcher/<id>.py`
  (real source files you can open and read; printed inline by Stage 1).
* **Factor DB entries** → `data/factors/showcase_factor_db.json`
  (seeded once from the real `factor_db.json`, then appended to). Generated
  factors are stored with status `candidate` — generated, **not yet validated**.
* **Strategy DB entries** → `data/strategies/showcase_strategy_db.json`
  (DRAFT strategies — designed, not yet back-tested).

> **Why strategies have no per-strategy code file (unlike factors).**
> By design, strategies are *not* generated as individual `.py` subclasses the
> way factors are. There are two generic strategy classes —
> `DynamicStrategy` (static-weight blends) and `ModelStrategy` (fitted ML) — and
> every strategy is an *instance* of one of them, fully described by its
> `StrategyRecord` in the strategy DB (model type, hyper-parameters, features,
> weights, holding period, …). `pipeline.strategy_from_record()` rebuilds the
> runnable object from that record. For an ML strategy the *trained model* is
> stored separately as a binary `.joblib` artifact under
> `data/strategies/models/` (reloaded via `ModelStrategy.from_artifact`), not as
> source code. In this showcase the fitting step is skipped, so ML strategies
> have **no** `.joblib` artifact — the record just holds the spec (the model that
> *would* be fit), which is why only static-weight strategies are instantiated
> for display.

To start completely fresh, delete the showcase copies (and, optionally, the
generated factor files):

```bash
rm -f data/factors/showcase_factor_db.json data/strategies/showcase_strategy_db.json
# generated factor subclasses (researcher/ is gitignored except __init__.py):
find quant_fund_agent/factors/researcher -name '*.py' ! -name '__init__.py' -delete
```

## How the "no data" mode works

The scripts set two environment variables (in `showcase_common.py`) before
importing any agent code:

* `QF_USE_MCP=0` — run every agent tool **in-process**, so you don't need to
  start any MCP servers.
* `FACTOR_DB_PATH=data/factors/showcase_factor_db.json` — redirect factor-DB
  reads and writes to the showcase copy.

No market-data files are ever opened, because the scripts stop before the
back-testing steps that would load them.
