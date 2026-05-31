"""Run the entire quant-fund agent pipeline once, end-to-end.

This is the "one command" view of the fund: it drives every agent stage
in the order they run in production and leaves behind a fully populated
set of databases (factors → strategies → portfolio).

    research (optional)  →  [ Selector → Architect → Statistician → persist ] × N
                                                                          ↓
                                                          Portfolio Manager rebalance

Each stage is a thin call into :mod:`quant_fund_agent.pipeline`, so this
script is really just an *orchestration policy* over reusable functions.

Relationship to the next milestone (2-month backtest)
-----------------------------------------------------
The backtest replaces the single linear run below with a scheduled loop
over 10-second-bar trading days:

    for day in trading_days:
        strategy_db.append_returns(... live PnL for `day` ...)   # mark-to-market
        if week_boundary(day):
            run_research_session(session_id=week_tag, cutoff_date=day)
            for _ in range(n_strategy_attempts):
                res = run_strategy_pipeline(cutoff_date=day)
                persist_approved_strategy(strategy_db, res)
            run_pm_rebalance(strategy_db, portfolio_db, committee=True)

i.e. the *same* pipeline functions this script calls, fired weekly instead
of once.  Nothing here needs to change for that — the persistence,
correlation caching, and PM committee paths are already incremental.

Examples
--------
::

    # Build a 3-strategy book, then run a 3-PM committee over it.
    python run_fund.py --n-strategies 3

    # Include a fresh factor-research session first.
    python run_fund.py --research --n-strategies 3

    # Single balanced PM instead of a committee; keep a wider universe.
    python run_fund.py --n-strategies 2 --no-committee --n-tickers 20

    # Resume: skip strategy research, just re-run the PM over the saved book.
    python run_fund.py --n-strategies 0
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from quant_fund_agent import pipeline
from quant_fund_agent.schemas import PMPersonality, VotingMethod

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_fund")

RULE = "=" * 80


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--research", action="store_true",
                   help="Run a Factor Researcher session before strategy work.")
    p.add_argument("--n-strategies", type=int, default=3,
                   help="How many strategy-research passes to attempt "
                        "(each may or may not yield an approved strategy).")
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Architect refinement iterations per strategy.")
    p.add_argument("--oos-ratio", type=float, default=0.2,
                   help="Fraction of data held out for the statistician.")
    p.add_argument("--n-tickers", type=int, default=10,
                   help="Universe cap for the architect/statistician backtests "
                        "(sets ARCHITECT_N_TICKERS).  0 = full universe.")
    p.add_argument("--no-committee", action="store_true",
                   help="Use a single balanced PM instead of a 3-PM committee.")
    p.add_argument("--voting",
                   choices=["simple_average", "weighted_average", "llm_moderator"],
                   default="simple_average",
                   help="Committee aggregation method.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing strategy/portfolio DBs and start "
                        "from an empty book.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")

    # Universe cap must be set before the architect module is imported.
    if args.n_tickers and args.n_tickers > 0:
        os.environ.setdefault("ARCHITECT_N_TICKERS", str(args.n_tickers))

    from quant_fund_agent.factors import discover_factors
    discover_factors()

    # ── Databases ────────────────────────────────────────────────────
    if args.fresh:
        from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
        strategy_db, portfolio_db = StrategyDatabase(), PortfolioDatabase()
    else:
        strategy_db, portfolio_db = pipeline.load_dbs()
    print(f"\nStarting book: {len(strategy_db.list_strategies())} strategies "
          f"already persisted.\n")

    # ── Stage 1: optional factor research ────────────────────────────
    if args.research:
        print(RULE)
        print("STAGE 1 · FACTOR RESEARCHER")
        print(RULE)
        research = pipeline.run_research_session(n_tickers=args.n_tickers or 15)
        print(f"  kept     : {research.get('kept_factor_ids', [])}")
        print(f"  rejected : {research.get('rejected_factor_ids', [])}")

    # ── Stage 2: strategy research × N (Selector → Architect → Stat) ──
    print("\n" + RULE)
    print(f"STAGE 2 · STRATEGY RESEARCH  (attempts: {args.n_strategies})")
    print(RULE)
    persisted = []
    for i in range(args.n_strategies):
        print(f"\n--- attempt {i + 1}/{args.n_strategies} ---")
        res = pipeline.run_strategy_pipeline(
            max_iterations=args.max_iterations,
            oos_ratio=args.oos_ratio,
        )
        if not res.approved:
            print(f"  rejected at the {res.reject_stage} stage.")
            continue
        record = pipeline.persist_approved_strategy(strategy_db, res)
        persisted.append(record)
        m = record.backtest_metrics
        oos = record.oos_backtest_metrics
        print(f"  APPROVED → {record.id}  ({record.name})")
        print(f"    factors  : {record.factor_ids}")
        print(f"    IS  Sharpe={getattr(m, 'sharpe_ratio', None)}  "
              f"MaxDD={getattr(m, 'max_drawdown', None)}")
        if oos:
            print(f"    OOS Sharpe={oos.sharpe_ratio}  MaxDD={oos.max_drawdown}")

    n_book = len(strategy_db.list_strategies())
    print(f"\nStrategy book now holds {n_book} strategies "
          f"({len(persisted)} added this run).")

    # ── Stage 3: Portfolio Manager rebalance ─────────────────────────
    print("\n" + RULE)
    print("STAGE 3 · PORTFOLIO MANAGER")
    print(RULE)
    if n_book == 0:
        print("  No strategies in the book — skipping the PM.")
        pipeline.save_dbs(strategy_db, portfolio_db)
        return

    use_committee = not args.no_committee and n_book >= 1
    record = pipeline.run_pm_rebalance(
        strategy_db, portfolio_db,
        personalities=[PMPersonality.DEFENSIVE, PMPersonality.BALANCED,
                       PMPersonality.AGGRESSIVE],
        committee=use_committee,
        voting_method=VotingMethod(args.voting),
        pm_name="fund_committee" if use_committee else "fund_pm",
    )

    if record is None:
        print("  PM produced no allocation.")
    else:
        live = [a for a in record.allocations if a.enabled]
        print(f"\n  Portfolio {record.id}")
        print(f"  PM / mode          : {record.pm_name} / {record.mode.value}")
        print(f"  Construction method: {record.construction_method.value}")
        print(f"  Allocations ({len(live)}):")
        for a in live:
            print(f"    {a.strategy_id:<40s} w = {a.weight:+.4f}")
        if record.flagged_strategy_ids:
            print(f"  Flagged : {record.flagged_strategy_ids}")
        if record.retired_strategy_ids:
            print(f"  Retired : {record.retired_strategy_ids}")
        if record.expected_metrics:
            print("  Expected metrics:")
            for k, v in record.expected_metrics.items():
                print(f"    {k:28s} {v}")

    # ── Persist everything ───────────────────────────────────────────
    pipeline.save_dbs(strategy_db, portfolio_db)
    print(f"\nSaved → {pipeline.STRATEGY_DB_PATH}, {pipeline.PORTFOLIO_DB_PATH}")


if __name__ == "__main__":
    main()
