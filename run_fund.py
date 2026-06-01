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
import json
import logging
import os
import pathlib

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


def _fmt(x, nd: int = 3) -> str:
    """Format a float KPI compactly; '-' for missing."""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


SHOWCASE_PATH = "data/strategies/showcase.json"


def _showcase_entry(res, record) -> dict:
    """Lightweight, JSON-able record of one strategy attempt for the notebook."""
    spec = res.strategy_spec
    arch = res.architect_result or {}
    sel = res.selector_result or {}

    trials = []
    for t in arch.get("trial_history", []):
        m = t.metrics or {}
        diag = m.get("diagnostics") or {}
        trials.append({
            "iteration": t.iteration,
            "model_type": t.spec.model_type,
            "model_params": t.spec.model_params,
            "n_features": len(t.spec.factor_ids or t.spec.weights),
            "valid_ic": diag.get("valid_ic"),
            "train_r2": diag.get("train_r2"),
            "valid_r2": diag.get("valid_r2"),
            "is_sharpe": m.get("sharpe_ratio"),
        })

    stat_tests = []
    for r in (res.stat_result or {}).get("test_results", []):
        stat_tests.append({
            "test_id": getattr(r, "test_id", None),
            "verdict": getattr(getattr(r, "verdict", None), "value", None),
            "value": getattr(r, "value", None),
            "threshold": getattr(r, "threshold", None),
            "summary": getattr(r, "summary", ""),
        })

    entry = {
        "approved": res.approved,
        "reject_stage": res.reject_stage,
        "hypothesis": res.hypothesis,
        "selection_rationale": sel.get("selection_rationale", ""),
        "model_type": spec.model_type if spec else None,
        "model_params": spec.model_params if spec else {},
        "factor_ids": list(spec.factor_ids) if spec else [],
        "target_horizon": spec.target_horizon if spec else None,
        "model_artifact_path": spec.model_artifact_path if spec else "",
        "holding_period": spec.holding_period if spec else None,
        "max_positions": spec.max_positions if spec else None,
        "min_conviction": spec.min_conviction if spec else None,
        "equal_weight": spec.equal_weight if spec else None,
        "trials": trials,
        "stat_tests": stat_tests,
        "final_decision": (res.stat_result or {}).get("final_decision"),
    }
    if record is not None:
        m, oos = record.backtest_metrics, record.oos_backtest_metrics
        entry.update({
            "id": record.id,
            "name": record.name,
            "artifact_path": (record.model_params or {}).get("model_artifact_path"),
            "is_sharpe": getattr(m, "sharpe_ratio", None),
            "oos_sharpe": getattr(oos, "sharpe_ratio", None),
            "deflated_sharpe": (record.metadata or {}).get("deflated_sharpe_ratio"),
        })
    return entry


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
    p.add_argument("--target-horizon", type=int, default=6,
                   help="Forecast horizon in bars: 1=10s, 6=1min (default), "
                        "60=10min.  The model predicts and holds at this horizon.")
    p.add_argument("--n-tickers", type=int, default=0,
                   help="Universe cap for the architect/statistician backtests "
                        "(sets ARCHITECT_N_TICKERS).  0 = full universe (default).")
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
    showcase = []
    for i in range(args.n_strategies):
        print(f"\n--- attempt {i + 1}/{args.n_strategies} ---")
        res = pipeline.run_strategy_pipeline(
            max_iterations=args.max_iterations,
            oos_ratio=args.oos_ratio,
            target_horizon=args.target_horizon,
        )
        if not res.approved:
            print(f"  rejected at the {res.reject_stage} stage.")
            showcase.append(_showcase_entry(res, None))
            continue
        record = pipeline.persist_approved_strategy(strategy_db, res)
        persisted.append(record)
        showcase.append(_showcase_entry(res, record))
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

    if persisted:
        print("\n" + RULE)
        print("ACCEPTED STRATEGIES (this run)")
        print(RULE)
        hdr = (f"  {'strategy':<30s} {'model':<16s} {'horizon':>7s} "
               f"{'IS Sharpe':>10s} {'OOS Sharpe':>11s} {'DSR':>6s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for rec in persisted:
            m, oos = rec.backtest_metrics, rec.oos_backtest_metrics
            dsr = (rec.metadata or {}).get("deflated_sharpe_ratio")
            th = (rec.model_params or {}).get("target_horizon", "-")
            print(f"  {rec.id[:29]:<30s} {rec.model_type:<16s} {str(th):>7s} "
                  f"{_fmt(getattr(m, 'sharpe_ratio', None)):>10s} "
                  f"{_fmt(getattr(oos, 'sharpe_ratio', None)):>11s} "
                  f"{_fmt(dsr):>6s}")

    # Persist a lightweight showcase artifact for the notebook (cached mode).
    pathlib.Path(SHOWCASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SHOWCASE_PATH, "w") as fh:
        json.dump({
            "target_horizon": args.target_horizon,
            "n_tickers": args.n_tickers or "all",
            "gates": {"dsr_threshold": 0.75, "max_sharpe_decay": 0.85},
            "entries": showcase,
        }, fh, indent=2, default=str)
    print(f"\nShowcase details → {SHOWCASE_PATH}")

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
