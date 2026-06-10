# Onboarding & Configuration

> Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). How a fresh clone gets from
> `git clone` to a running fund. Sections tagged **(planned)** track the roadmap.

## TL;DR (target experience)

```bash
git clone <repo> && cd QuantFundAgent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # put your keys here (or none, for yfinance)
python -m quant_fund_agent.setup   # guided wizard → writes quant.config.yaml

./venv/bin/python run_fund.py --n-strategies 1   # runs on YOUR data
```

No LOBSTER files, no code edits.

## The wizard (`python -m quant_fund_agent.setup`) — (planned, Phase 3)

A deterministic, dependency-light CLI that:

1. **Detects available keys** in `.env` (`FMP_API_KEY`, `ALPHAVANTAGE_API_KEY`;
   yfinance needs none) and offers only providers you can actually use.
2. **Prompts** for:
   - provider (from the detected set),
   - asset class (equity now; crypto/FX later),
   - universe — a **preset** (`sp100`, `sp500`, `nasdaq100`) or a **custom**
     comma-separated ticker list,
   - date range (`start`, `end`),
   - frequency (`1d` now; intraday later).
3. **Validation fetch** — pulls a few bars for one symbol to confirm the key and
   symbols resolve before committing.
4. **Writes `quant.config.yaml`** (schema below).

### Optional `--assist` (planned, Phase 5)

`python -m quant_fund_agent.setup --assist` adds an LLM layer that turns free
text ("mid-cap US tech, last 2 years, daily") into a proposed config, which the
deterministic wizard then shows for confirmation before writing. The core wizard
always works without an LLM key.

## `quant.config.yaml` schema

```yaml
data:
  provider: yfinance            # lobster | yfinance | fmp | alphavantage
  asset_class: equity           # equity (crypto/FX later)
  frequency: 1d                 # 1d | 1min | 10s | …
  start: 2024-01-01
  end: 2026-01-01
  universe:
    preset: sp100               # OR omit and use `tickers:`
    # tickers: [AAPL, MSFT, NVDA, …]
  n_tickers: null               # optional cap (memory); null = all
  cache_dir: data/market        # parquet cache root

# Optional overrides (sane defaults otherwise)
annualization:
  periods_per_year: auto        # auto-infer from frequency; or an explicit int

run:
  # passthrough knobs for run_fund.py / run_backtest.py if desired
```

### Precedence
1. Explicit env vars (highest — preserves existing scripts/tests).
2. `quant.config.yaml`.
3. Built-in defaults (LOBSTER `ticker_data/`, OpenAI `gpt-4o-mini`, etc.).

So existing LOBSTER workflows keep running untouched if no config file is
present.

## Secrets (`.env`)

Only API keys live in `.env` (never in `quant.config.yaml`, which is meant to be
shareable/committable):

```
OPENAI_API_KEY=sk-...           # required for the LLM agents
FMP_API_KEY=...                 # optional
ALPHAVANTAGE_API_KEY=...        # optional
# yfinance: no key required
```

A `.env.example` enumerates every recognized key.

## Where the data lands

Fetched vendor data is cached as parquet under `cache_dir` (default
`data/market/`), keyed by `(provider, symbol, freq, asset_class)`. Re-runs read
the cache; only missing date ranges hit the API. This both amortizes rate limits
and makes a run reproducible.
