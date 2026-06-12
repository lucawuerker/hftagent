# QuantFundAgent

Master's thesis project (Mathematics & Finance, Imperial College London).

## Goal
Build LLM agents that operate like a professional quant fund: researching
features, designing strategies, and deploying them live.

## Architecture
A LangGraph multi-agent pipeline mirroring a quant fund's org chart:

- **Factor Researcher** – reads papers, brainstorms alpha ideas, generates
  factor code, backtests each one (recording its IC for reference) and keeps
  every factor that runs — IC magnitude is not a keep/drop gate.
- **Selector** – picks factors for a hypothesis.
- **Architect** – combines factors into a strategy via a refinement loop.
- **Statistician** – OOS tests, deflated Sharpe, accept/reject gate.
- **Portfolio Manager** – screens, allocates capital, monitors/retires
  strategies (single PM or a committee).

State flows through shared databases (factor / paper / strategy / portfolio).
The stage sequence lives in `quant_fund_agent/pipeline.py` so the notebook,
scripts, and backtests all drive the agents identically.

## Status
A working **MVP** of the full pipeline exists. Each stage and agent is
intended to be advanced significantly in future work.

**Research-LLM comparison (done).** Named factor-research *preruns*
(`run_factor_research.py --name <id> --model <llm> [--llm-provider <p>]
--dedup-scope prerun`) mine N factors with a chosen research model into a
self-contained `data/factors/preruns/<name>/`. `run_model_comparison.py` then
compares several preruns' factor sets on three axes — **single-factor IC**
(raw quality), **brute-force ML** (factors → model catalog + ensemble, OOS),
and **downstream agents** (the full Selector→Architect→Statistician→PM fund,
single-pass OOS) — emitting presentation-ready figures, CSV/JSON tables, a
`report.md` and a `comparison.ipynb` under `data/comparisons/<id>/`
(`quant_fund_agent/comparison/`). IC + brute-force are LLM-free; only the
downstream track and `--research` spend LLM. Factors needing fields the current
data lacks are filtered (and reported), so it runs on today's 51-ticker LOBSTER
sample now and re-runs unchanged once full LOBSTER / FMP data is downloaded.

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
