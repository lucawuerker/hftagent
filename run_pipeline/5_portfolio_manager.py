"""Stage 5: Portfolio Manager — allocate capital over the strategy book.

Loads the **real** strategy + portfolio databases, refreshes the cross-strategy
correlation matrix, and runs one PM (or a committee of PMs) to screen, size and
flag/retire strategies into a ``PortfolioRecord`` — which is written back to
``data/portfolio/portfolio_db.json``.

The PM needs no market-data panel: it works from the per-strategy returns series
already persisted on each ``StrategyRecord`` (so run ``4_statistician.py`` or
``run_pipeline.py`` first to populate the book).  It uses the correct intraday
annualisation factor (``pipeline.BARS_PER_YEAR``) so expected-Sharpe figures line
up with each strategy's annualised metrics.

Usage::

    # Single balanced PM (default construction method for the personality).
    ./venv/bin/python run_pipeline/5_portfolio_manager.py

    # Committee: 3 PMs propose, simple-average consensus.
    ./venv/bin/python run_pipeline/5_portfolio_manager.py --committee defensive,balanced,aggressive

    # Committee with an LLM moderator synthesising the consensus.
    ./venv/bin/python run_pipeline/5_portfolio_manager.py \
        --committee defensive,balanced,aggressive --voting llm_moderator

    # Active single PM (the LLM also picks the construction method).
    ./venv/bin/python run_pipeline/5_portfolio_manager.py --mode active --personality aggressive
"""

from __future__ import annotations

import argparse
import logging

import pipeline_common as pc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pm-name", default=None,
                   help="Name for this PM/committee (used in the audit log).")
    p.add_argument("--mode", choices=["selector", "active"], default="selector",
                   help="selector = fixed construction; active = the LLM picks "
                        "the construction method too.")
    p.add_argument("--personality",
                   choices=["defensive", "balanced", "aggressive"],
                   default="balanced",
                   help="Single-PM personality (ignored when --committee is set).")
    p.add_argument("--committee", default=None,
                   help="Comma-separated personalities to run as a committee, "
                        "e.g. 'defensive,balanced,aggressive'.")
    p.add_argument("--voting",
                   choices=["simple_average", "weighted_average", "llm_moderator"],
                   default="simple_average",
                   help="How a committee aggregates per-PM proposals.")
    p.add_argument("--method", default=None,
                   choices=["equal_weight", "inverse_volatility", "min_variance",
                            "mean_variance", "max_sharpe", "risk_parity",
                            "hierarchical_risk_parity"],
                   help="Override the construction method (SELECTOR mode only).")
    p.add_argument("--target-n", type=int, default=None,
                   help="Override the personality's target number of strategies.")
    pc.add_common_args(p, with_data=False)  # --mcp only (no market-data panel)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.banner("STAGE 5 · PORTFOLIO MANAGER")

    pc.require_openai_key()
    pc.configure_runtime(use_mcp=args.mcp)

    from quant_fund_agent import pipeline
    from quant_fund_agent.schemas import (
        ConstructionMethod,
        PMMode,
        PMPersonality,
        VotingMethod,
    )

    strategy_db, portfolio_db = pipeline.load_dbs()
    n = len(strategy_db.list_strategies())
    if n == 0:
        print(f"\n  Strategy book is empty ({pipeline.STRATEGY_DB_PATH}).")
        print("  Run ./venv/bin/python run_pipeline/4_statistician.py first.")
        return
    print(f"\n  Loaded {n} strategies from {pipeline.STRATEGY_DB_PATH}.")

    committee = bool(args.committee)
    if committee:
        personalities = [PMPersonality(p.strip())
                         for p in args.committee.split(",") if p.strip()]
        if len(personalities) < 2:
            raise SystemExit("--committee needs at least two personalities.")
        default_name = "fund_committee"
    else:
        personalities = [PMPersonality(args.personality)]
        default_name = "fund_pm"

    print(f"  PM(s): {[p.value for p in personalities]}  "
          f"mode={args.mode}  "
          + (f"voting={args.voting}" if committee else f"method={args.method or 'default'}"))

    record = pipeline.run_pm_rebalance(
        strategy_db, portfolio_db,
        personalities=personalities,
        mode=PMMode(args.mode),
        committee=committee,
        voting_method=VotingMethod(args.voting),
        pm_name=args.pm_name or default_name,
        construction_method=ConstructionMethod(args.method) if args.method else None,
        target_n_strategies=args.target_n,
    )

    if record is None:
        print("\n  PM produced no portfolio record.")
        return

    live = [a for a in record.allocations if a.enabled]
    pc.section("PORTFOLIO ALLOCATION")
    print(f"  Portfolio ID        : {record.id}")
    print(f"  PM / mode           : {record.pm_name} / {record.mode.value}")
    print(f"  Construction method : {record.construction_method.value}")
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
    if record.pm_rationale:
        pc.section("RATIONALE")
        print(record.pm_rationale)

    pipeline.save_dbs(strategy_db, portfolio_db)
    pc.banner("STAGE 5 COMPLETE")
    print(f"  Saved → {pipeline.PORTFOLIO_DB_PATH}, {pipeline.STRATEGY_DB_PATH}")


if __name__ == "__main__":
    main()
