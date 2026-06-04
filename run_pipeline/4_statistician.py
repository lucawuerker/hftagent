"""Stage 4: Statistician Agent — out-of-sample acceptance, then persist.

Runs the full strategy-research pass on the local market data — Selector →
Architect (fit + backtest) → **Statistician** — and applies the accept/reject
gate.  The Statistician re-fits the approved design on the in-sample window and
evaluates it out-of-sample (deflated Sharpe, OOS Sharpe decay, …).

If the strategy is accepted it is persisted into the **real** strategy database
(``data/strategies/strategy_db.json``) as a ``StrategyRecord`` with its in-sample
+ out-of-sample metrics and per-bar returns series — this is the step that
populates the strategy book the Portfolio Manager allocates over.

Stages 3 and 4 each re-run the Selector + Architect (the per-step scripts are
intentionally self-contained).  Use ``run_pipeline.py`` for one chained pass.

Usage::

    ./venv/bin/python run_pipeline/4_statistician.py --n-tickers 8
    ./venv/bin/python run_pipeline/4_statistician.py --no-persist   # don't touch the book
    ./venv/bin/python run_pipeline/4_statistician.py --fresh        # start from an empty book
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
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Architect refinement iterations.")
    p.add_argument("--oos-ratio", type=float, default=0.2,
                   help="Fraction of data held out for the Statistician.")
    p.add_argument("--target-horizon", type=int, default=6,
                   help="Forecast horizon in 10-sec bars (1=10s, 6=1min, 60=10min).")
    p.add_argument("--no-persist", action="store_true",
                   help="Run the gate but do not write to the strategy DB.")
    p.add_argument("--fresh", action="store_true",
                   help="Start from an empty strategy book instead of the saved one.")
    pc.add_common_args(p)  # --mcp, --n-tickers, --data-dir
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.banner("STAGE 4 · STATISTICIAN")
    print("Selector → Architect → Statistician.  Out-of-sample accept/reject gate,")
    print("then persist any accepted strategy into the real strategy book.")

    pc.require_openai_key()
    pc.require_market_data(args.data_dir)
    pc.configure_runtime(n_tickers=args.n_tickers, data_dir=args.data_dir,
                         use_mcp=args.mcp)

    from quant_fund_agent import pipeline
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase

    res = pipeline.run_strategy_pipeline(
        max_iterations=args.max_iterations,
        oos_ratio=args.oos_ratio,
        target_horizon=args.target_horizon,
        run_statistician=True,
    )

    pc.section("STRATEGY UNDER TEST")
    print(f"  Hypothesis : {res.hypothesis}")
    spec = res.strategy_spec
    if spec is not None:
        print(f"  Strategy   : {spec.strategy_name}  ({spec.model_type})")
        print(f"  Factors    : {spec.factor_ids or list(spec.weights)}")

    if res.reject_stage == "architect":
        pc.banner("STAGE 4 COMPLETE — rejected by the Architect")
        print("  The Architect never approved a design, so the Statistician did")
        print("  not run.  Nothing persisted.")
        return

    stat = res.stat_result or {}
    pc.section("STATISTICAL TESTS")
    for r in stat.get("test_results", []):
        v = r.verdict.value.upper()
        print(f"\n  [{v:4s}] {r.test_name}")
        print(f"    value={pc.fmt(r.value, 4)}  threshold={pc.fmt(r.threshold, 4)}  "
              f"p={pc.fmt(r.p_value, 4)}")
        if r.summary:
            print(f"    {r.summary}")

    if stat.get("risk_assessment"):
        pc.section("RISK ASSESSMENT")
        print(stat["risk_assessment"])

    pc.section(f"FINAL VERDICT: {str(stat.get('final_decision', '?')).upper()}")
    print(stat.get("final_reasoning_statistician", "(none)"))

    # ── Persist an accepted strategy into the real book ──────────────────
    if not res.approved:
        pc.banner("STAGE 4 COMPLETE — strategy rejected")
        print("  Nothing persisted.")
        return

    if args.no_persist:
        pc.banner("STAGE 4 COMPLETE — accepted (not persisted, --no-persist)")
        return

    if args.fresh:
        strategy_db, portfolio_db = StrategyDatabase(), PortfolioDatabase()
    else:
        strategy_db, portfolio_db = pipeline.load_dbs()

    record = pipeline.persist_approved_strategy(strategy_db, res)
    pipeline.save_dbs(strategy_db, portfolio_db)

    pc.banner("STAGE 4 COMPLETE — strategy accepted + persisted")
    if record is not None:
        m, oos = record.backtest_metrics, record.oos_backtest_metrics
        print(f"  {record.id}  ({record.name})")
        print(f"    IS  Sharpe = {pc.fmt(getattr(m, 'sharpe_ratio', None))}   "
              f"OOS Sharpe = {pc.fmt(getattr(oos, 'sharpe_ratio', None))}")
        print(f"  Book now holds {len(strategy_db.list_strategies())} strategies.")
    print("  Next: ./venv/bin/python run_pipeline/5_portfolio_manager.py")


if __name__ == "__main__":
    main()
