# QuantFundAgent

A LangGraph-based multi-agent system that mirrors the structure of a quantitative fund / prop trading firm.

## Architecture

```
                     ┌──────────────────────┐
                     │     Orchestrator      │
                     │  (routes to agents)   │
                     └──────┬───┬───┬───────┘
                            │   │   │
            ┌───────────────┘   │   └───────────────┐
            ▼                   ▼                   ▼
  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
  │  Factor Research │ │    Strategy     │ │    Portfolio     │
  │      Agent       │ │     Agent       │ │    Manager       │
  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
           │                   │                   │
           ▼                   ▼                   ▼
  hypothesis → factor   select factors →    review strategies →
  → backtest → eval     design strategy →   assess correlations →
  → add to DB           backtest → eval →   allocate portfolio →
                        add to DB           flag underperformers
           │                   │                   │
           ▼                   ▼                   ▼
  ┌─────────────────────────────────────────────────────────┐
  │                   Shared Databases                      │
  │  ┌──────────┐   ┌──────────┐   ┌──────────────┐        │
  │  │ Factor DB│   │ Paper DB │   │ Strategy DB  │        │
  │  └──────────┘   └──────────┘   └──────────────┘        │
  └─────────────────────────────────────────────────────────┘
```

## Project Structure

```
QuantFundAgent/
├── quant_fund_agent/
│   ├── schemas.py                  # Pydantic records: FactorRecord, Paper, StrategyRecord, …
│   ├── state.py                    # LangGraph state definitions (FundState + per-agent)
│   ├── orchestrator.py             # Top-level graph that routes to agent subgraphs
│   ├── main.py                     # Entry point
│   │
│   ├── databases/                  # Metadata registries (persist to JSON)
│   │   ├── factor_db.py            # FactorRecord registry + correlation matrix
│   │   ├── paper_db.py             # Paper registry with lookahead-bias filtering
│   │   └── strategy_db.py          # StrategyRecord registry
│   │
│   ├── factors/                    # Factor implementations (Python classes)
│   │   ├── base.py                 # BaseFactor ABC — every factor inherits this
│   │   ├── registry.py             # @register_factor decorator + lookup functions
│   │   ├── _discover.py            # Auto-imports all factor modules at startup
│   │   ├── momentum/
│   │   │   └── three_soldiers.py   # Example: Three White Soldiers candlestick pattern
│   │   └── mean_reversion/
│   │       └── rsi_mean_reversion.py  # Example: RSI-based mean-reversion signal
│   │
│   ├── strategies/                 # Strategy implementations (Python classes)
│   │   ├── base.py                 # BaseStrategy ABC — every strategy inherits this
│   │   ├── registry.py             # @register_strategy decorator + lookup functions
│   │   ├── _discover.py            # Auto-imports all strategy modules at startup
│   │   └── implementations/
│   │       └── momentum_reversion_combo.py  # Example: blended momentum + reversion
│   │
│   ├── agents/                     # LangGraph subgraphs
│   │   ├── factor_research/graph.py
│   │   ├── strategy/graph.py
│   │   └── portfolio_manager/graph.py
│   │
│   ├── backtesting/
│   │   └── engine.py               # Backtest stubs (single-factor & strategy)
│   └── utils/
│
├── data/
│   ├── factors/                    # Serialised factor metadata
│   ├── papers/
│   │   ├── index.json              # Paper metadata (loaded by PaperDatabase)
│   │   └── pdfs/                   # Actual PDF files
│   └── strategies/                 # Serialised strategy metadata
│
├── requirements.txt
├── .env.example
└── README.md
```

## Key Design Decisions

### Factor & Strategy as Code

Every factor is a Python class that inherits from `BaseFactor` and implements a `calc(data)` method.  Strategies inherit from `BaseStrategy` and implement `calc(factor_signals, data)`.  This keeps computation in version-controlled code while metadata (backtest results, status, trading-idea links) lives in the database records (`FactorRecord` / `StrategyRecord` in `schemas.py`).

The `class_name` field on each database record links it back to the implementation class:

```
FactorRecord (schemas.py)          BaseFactor subclass (factors/)
┌──────────────────────┐           ┌──────────────────────────┐
│ id: "rsi_mr"         │──────────▶│ RSIMeanReversionSignal   │
│ class_name:          │  registry │   factor_id = "rsi_mr"   │
│   "RSIMeanReversion" │  lookup   │   def calc(self, data)   │
│ status: "approved"   │           └──────────────────────────┘
│ backtest_metrics: …  │
└──────────────────────┘
```

### Factor Organisation

Factors live under `factors/<category>/`, one file per factor:

```
factors/
├── momentum/
│   ├── three_soldiers.py
│   └── moving_average_cross.py     ← add more here
├── mean_reversion/
│   └── rsi_mean_reversion.py
├── volatility/                     ← add new categories as directories
└── …
```

At startup call `discover_factors()` to auto-import every module and populate the global registry.

### Paper Storage & Lookahead-Bias Prevention

Papers are stored in `data/papers/`:
- **`index.json`** — array of `Paper` objects with a mandatory `published_date` field.
- **`pdfs/`** — the actual PDF files, referenced by `file_path`.

`PaperDatabase.list_papers_before(cutoff_date)` returns only papers published before the cutoff.  Use this whenever the factor research agent reads papers for a backtest period to avoid lookahead bias.

### Registry Pattern

```python
from quant_fund_agent.factors import discover_factors, instantiate_factor

discover_factors()                          # imports all factor modules
factor = instantiate_factor("rsi_mean_reversion")  # returns RSIMeanReversionSignal()
signal = factor.calc(data)
```

## Getting Started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python -m quant_fund_agent.main
```
