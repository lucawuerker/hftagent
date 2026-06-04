"""Run the whole quant-fund pipeline once, end-to-end, on the local market data.

This is the "one command" driver: it fires every agent stage in production order
and leaves behind a fully populated set of **real** databases (factors →
strategies → portfolio).

    [ Factor Researcher ]   →   [ Selector → Architect → Statistician → persist ] × N
                                                                            ↓
                                                            [ Portfolio Manager rebalance ]

Each stage is a thin call into :mod:`quant_fund_agent.pipeline` (the same seam the
notebook, ``run_fund.py`` and the walk-forward backtest use), so this script is
just an *orchestration policy* over reusable functions.  Any stage can be skipped.

Examples
--------
::

    # Everything: research → build a 3-strategy book → run a 3-PM committee.
    ./venv/bin/python run_pipeline/run_pipeline.py --n-tickers 8

    # Skip the (slow) factor-research stage; reuse the existing factor catalog.
    ./venv/bin/python run_pipeline/run_pipeline.py --skip-research --n-strategies 2

    # Just re-run the PM over the already-persisted strategy book.
    ./venv/bin/python run_pipeline/run_pipeline.py --skip-research --skip-strategy

    # Start from an empty book and run the MCP server path.
    ./venv/bin/python run_pipeline/run_pipeline.py --fresh --mcp --n-tickers 8
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
    # stage skips
    p.add_argument("--skip-research", action="store_true",
                   help="Skip the Factor Researcher stage.")
    p.add_argument("--skip-strategy", action="store_true",
                   help="Skip the Selector → Architect → Statistician stage.")
    p.add_argument("--skip-pm", action="store_true",
                   help="Skip the Portfolio Manager rebalance stage.")
    # research knobs
    p.add_argument("--n-papers", type=int, default=2)
    p.add_argument("--n-ideas", type=int, default=3)
    p.add_argument("--horizon", type=int, default=6,
                   help="Horizon (10-sec bars) at which the researcher records "
                        "each factor's IC for reference (not a keep/drop gate).")
    # strategy knobs
    p.add_argument("--n-strategies", type=int, default=3,
                   help="Strategy-research passes to attempt (each may or may "
                        "not yield an accepted strategy).")
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Architect refinement iterations per strategy.")
    p.add_argument("--oos-ratio", type=float, default=0.2)
    p.add_argument("--target-horizon", type=int, default=6)
    # PM knobs
    p.add_argument("--no-committee", action="store_true",
                   help="Use a single balanced PM instead of a 3-PM committee.")
    p.add_argument("--voting",
                   choices=["simple_average", "weighted_average", "llm_moderator"],
                   default="simple_average")
    # run-wide
    p.add_argument("--fresh", action="store_true",
                   help="Ignore the saved strategy/portfolio DBs; start empty.")
    pc.add_common_args(p)  # --mcp, --n-tickers, --data-dir
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.require_openai_key()
    pc.require_market_data(args.data_dir)
    pc.configure_runtime(n_tickers=args.n_tickers, data_dir=args.data_dir,
                         use_mcp=args.mcp)

    from quant_fund_agent import pipeline
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.schemas import PMPersonality, VotingMethod

    discover_factors()

    if args.fresh:
        strategy_db, portfolio_db = StrategyDatabase(), PortfolioDatabase()
    else:
        strategy_db, portfolio_db = pipeline.load_dbs()
    print(f"\nStarting book: {len(strategy_db.list_strategies())} strategies "
          f"already persisted.")

    # ── Stage 1: Factor Researcher ───────────────────────────────────────
    if args.skip_research:
        pc.banner("STAGE 1 · FACTOR RESEARCHER  (skipped)")
    else:
        pc.banner("STAGE 1 · FACTOR RESEARCHER")
        research = pipeline.run_research_session(
            n_papers=args.n_papers,
            n_ideas=args.n_ideas,
            horizon=args.horizon,
            n_tickers=(args.n_tickers if args.n_tickers > 0 else None),
        )
        print(f"  kept     : {research.get('kept_factor_ids', [])}")
        print(f"  rejected : {research.get('rejected_factor_ids', [])}")

    # ── Stage 2: Strategy research × N ───────────────────────────────────
    persisted = []
    if args.skip_strategy:
        pc.banner("STAGE 2 · STRATEGY RESEARCH  (skipped)")
    else:
        pc.banner(f"STAGE 2 · STRATEGY RESEARCH  (attempts: {args.n_strategies})")
        for i in range(args.n_strategies):
            print(f"\n--- attempt {i + 1}/{args.n_strategies} ---")
            res = pipeline.run_strategy_pipeline(
                max_iterations=args.max_iterations,
                oos_ratio=args.oos_ratio,
                target_horizon=args.target_horizon,
            )
            if not res.approved:
                print(f"  rejected at the {res.reject_stage} stage.")
                continue
            record = pipeline.persist_approved_strategy(strategy_db, res)
            persisted.append(record)
            m, oos = record.backtest_metrics, record.oos_backtest_metrics
            print(f"  APPROVED → {record.id}  ({record.name})")
            print(f"    IS Sharpe={pc.fmt(getattr(m, 'sharpe_ratio', None))}  "
                  f"OOS Sharpe={pc.fmt(getattr(oos, 'sharpe_ratio', None))}")

    n_book = len(strategy_db.list_strategies())
    print(f"\nStrategy book now holds {n_book} strategies "
          f"({len(persisted)} added this run).")

    if persisted:
        pc.section("ACCEPTED STRATEGIES (this run)")
        hdr = (f"  {'strategy':<30s} {'model':<16s} {'horizon':>7s} "
               f"{'IS Sharpe':>10s} {'OOS Sharpe':>11s} {'DSR':>6s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rec in persisted:
            m, oos = rec.backtest_metrics, rec.oos_backtest_metrics
            dsr = (rec.metadata or {}).get("deflated_sharpe_ratio")
            th = (rec.model_params or {}).get("target_horizon", "-")
            print(f"  {rec.id[:29]:<30s} {rec.model_type:<16s} {str(th):>7s} "
                  f"{pc.fmt(getattr(m, 'sharpe_ratio', None)):>10s} "
                  f"{pc.fmt(getattr(oos, 'sharpe_ratio', None)):>11s} "
                  f"{pc.fmt(dsr):>6s}")

    # ── Stage 3: Portfolio Manager rebalance ─────────────────────────────
    if args.skip_pm:
        pc.banner("STAGE 3 · PORTFOLIO MANAGER  (skipped)")
    elif n_book == 0:
        pc.banner("STAGE 3 · PORTFOLIO MANAGER")
        print("  No strategies in the book — skipping the PM.")
    else:
        pc.banner("STAGE 3 · PORTFOLIO MANAGER")
        use_committee = not args.no_committee
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
            print(f"\n  Portfolio {record.id}  ({record.pm_name})")
            print(f"  Construction method: {record.construction_method.value}")
            print(f"  Allocations ({len(live)}):")
            for a in live:
                print(f"    {a.strategy_id:<40s} w = {a.weight:+.4f}")
            if record.flagged_strategy_ids:
                print(f"  Flagged : {record.flagged_strategy_ids}")
            if record.retired_strategy_ids:
                print(f"  Retired : {record.retired_strategy_ids}")

    # ── Persist everything ───────────────────────────────────────────────
    pipeline.save_dbs(strategy_db, portfolio_db)
    pc.banner("PIPELINE COMPLETE")
    print(f"  Saved → {pipeline.STRATEGY_DB_PATH}, {pipeline.PORTFOLIO_DB_PATH}")


if __name__ == "__main__":
    main()
