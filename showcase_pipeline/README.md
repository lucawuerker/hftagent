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
| 4. Portfolio Manager | `4_portfolio_manager.py` | monitors live strategies → screens the book → allocates capital → persists the decision | — (runs **fully**; see the note below) |

The **Statistician** stage is intentionally not included here: it exists only
to *validate* strategies using the market-data back-tests, so there is nothing
meaningful for it to do without the data.

The **Portfolio Manager** (Stage 4) is included, but it works differently from
Stages 1–3.  The PM has no creative step that *precedes* the data — its whole
job is to allocate capital across strategies that have **already been
backtested and approved**, reasoning over each strategy's summary metrics and
its return series (to estimate cross-strategy correlations).  Crucially, the PM
**never opens the order-book / price panel** at all: those two inputs already
live in the repo — the strategy database (`data/strategies/strategy_db.json`)
and the per-strategy return CSVs under `data/strategies/returns/`.  So by
default Stage 4 runs the **real PM agent, end-to-end and unmodified, over the
real strategy book** (the approved strategies your earlier pipeline produced),
with no market data.

In its default **SELECTOR** mode the PM is fully deterministic and needs **no
LLM and no network at all** (monitoring uses the personality's rule-based
thresholds; screening and construction are pure quant maths).  `--mode active`
(or `--committee --voting llm_moderator`) instead lets an LLM drive the
monitor / screen / construction-method choices, which needs an `OPENAI_API_KEY`.

The real book is a set of strong, healthy strategies, so the **monitor** node
keeps them all.  To also watch the monitor **flag / retire** degraded
strategies, pass `--synthetic`: it swaps in a hand-built book that includes a
drifting strategy and a broken one (their metrics are hand-set and their return
series generated with a controlled correlation structure — still no market
data).  Either way every *agent node* is the real one; only the book differs.

Stage 4 only ever **reads** `strategy_db.json` — the allocation it writes goes
to *separate showcase copies* (see below), so your real databases stay clean.

## Prerequisites

* The project virtual environment.
* An OpenAI API key — the agents call an LLM to brainstorm, select and design.
  Put it in a `.env` file at the repository root:

  ```
  OPENAI_API_KEY=sk-...
  ```

  Stages 1–3 always need this.  Stage 4 (Portfolio Manager) needs it **only**
  for its LLM-driven variants (`--mode active`, `--committee --voting
  llm_moderator`); the default SELECTOR mode is deterministic and runs with no
  key and no network.

## Running

From anywhere in the repo (the scripts `cd` to the project root themselves):

```bash
./venv/bin/python showcase_pipeline/1_factor_research.py
./venv/bin/python showcase_pipeline/2_selector.py
./venv/bin/python showcase_pipeline/3_architect.py
./venv/bin/python showcase_pipeline/4_portfolio_manager.py
```

Run Stages 1–3 in order for a coherent story (the factors created in Stage 1
become selectable in Stage 2, whose hypothesis feeds Stage 3), or run any one on
its own — each is self-contained.  Stage 4 reads the **approved** strategies in
`strategy_db.json` directly (the PM needs validated, backtested strategies — not
the unvalidated candidates / DRAFTs Stages 1–3 produce), so it stands on its own.

Some Stage 4 variations to try:

```bash
# Different risk appetites (each picks a different roster + construction method):
./venv/bin/python showcase_pipeline/4_portfolio_manager.py --personality defensive
./venv/bin/python showcase_pipeline/4_portfolio_manager.py --personality aggressive

# A 3-PM committee (defensive + balanced + aggressive) voting on one book:
./venv/bin/python showcase_pipeline/4_portfolio_manager.py --committee

# Swap the real book for a degraded one to watch the monitor flag/retire:
./venv/bin/python showcase_pipeline/4_portfolio_manager.py --synthetic

# LLM-driven instead of rule-based (needs OPENAI_API_KEY):
./venv/bin/python showcase_pipeline/4_portfolio_manager.py --mode active
```

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
* **Stage 4 (PM) inputs** are *read-only*: the real `data/strategies/strategy_db.json`
  and its return CSVs under `data/strategies/returns/` are never modified.
  (With `--synthetic` the book is instead generated into
  `data/strategies/showcase_pm_strategy_db.json` + `data/strategies/showcase_pm_returns/`,
  rebuilt deterministically on every run.)
* **Stage 4 (PM) outputs** →
  * `data/portfolio/showcase_pm_portfolio_db.json` — the persisted
    `PortfolioRecord` (weights, flagged/retired strategies, ex-ante metrics,
    rationale).
  * `data/strategies/showcase_pm_strategy_db.json` — a *copy* of the strategy DB
    after the rebalance, so you can see each strategy's updated `pm_status`
    (LIVE / PAUSED / FLAGGED / RETIRED) without touching the real DB.

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

Stage 4's output files (`showcase_pm_portfolio_db.json`,
`showcase_pm_strategy_db.json`, and — with `--synthetic` — `showcase_pm_returns/`)
are overwritten on every run, so there's nothing to reset — just re-run the
script.

## How the "no data" mode works

The scripts set two environment variables (in `showcase_common.py`) before
importing any agent code:

* `QF_USE_MCP=0` — run every agent tool **in-process**, so you don't need to
  start any MCP servers (this also covers the PM's quant-portfolio tools in
  Stage 4).
* `FACTOR_DB_PATH=data/factors/showcase_factor_db.json` — redirect factor-DB
  reads and writes to the showcase copy.

No market-data files are ever opened.  For Stages 1–3 that's because the
scripts stop before the back-testing steps that would load them; for Stage 4
it's because the PM never touches the market-data panel at all — it reasons
purely over the strategy book (the strategy DB + the persisted per-strategy
return series).
