"""Run a walk-forward backtest of the whole fund.

This is the "through time" analogue of ``run_fund.py``: instead of one linear
research → strategy → PM pass, it steps the fund over a date range on a weekly
grid, holding **research / strategy / PM meetings** on a schedule and trading the
resulting book on genuinely unseen data with realistic (spread-aware) execution.

The backtest is deliberately separate from the agents — it drives them only via
``quant_fund_agent.pipeline`` — so the same agent code path runs in live trading.

Examples
--------
::

    # Two months, weekly meetings, consolidated netted book.
    python run_backtest.py --start 2019-01-02 --end 2019-03-02

    # Smoke run: shrink warm-up so meetings fire on a short span; small universe.
    QF_USE_MCP=0 ARCHITECT_N_TICKERS=8 \
      python run_backtest.py --start 2019-01-02 --end 2019-02-15 \
      --warmup 2W --initial-strategies 2 --n-strategies 1 --execution netted

    # Compare the independent-pod model on the same span.
    python run_backtest.py --start 2019-01-02 --end 2019-03-02 --execution pod
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from quant_fund_agent.schemas import PMPersonality
from quant_fund_agent.simulation import BacktestConfig, run_backtest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_backtest")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # span / schedule
    p.add_argument("--start", required=True, help="ISO start date (inclusive).")
    p.add_argument("--end", required=True, help="ISO end date (exclusive).")
    p.add_argument("--warmup", default="1M",
                   help="History required before the first meeting (e.g. 1M, 2W).")
    p.add_argument("--grid-freq", default="1W", help="Rebalance-grid step.")
    p.add_argument("--research-every", default="1W")
    p.add_argument("--strategy-every", default="1W")
    p.add_argument("--pm-every", default="1W")
    p.add_argument("--no-research", action="store_true",
                   help="Skip factor-research meetings entirely.")
    # strategy counts
    p.add_argument("--initial-strategies", type=int, default=5,
                   help="Strategies built at the FIRST strategy meeting (bootstrap).")
    p.add_argument("--n-strategies", type=int, default=2,
                   help="Strategies built at every strategy meeting after the first.")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--oos-ratio", type=float, default=0.2)
    p.add_argument("--target-horizon", type=int, default=6)
    # execution / costs
    p.add_argument("--execution", choices=["netted", "pod"], default="netted",
                   help="Consolidated netted fund book (default) or independent pods.")
    p.add_argument("--fund-max-positions", type=int, default=50)
    p.add_argument("--max-weight-per-name", type=float, default=0.10)
    p.add_argument("--gross-leverage", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.5)
    p.add_argument("--no-spread-cost", action="store_true",
                   help="Disable the spread component of costs (commission only).")
    # PM
    p.add_argument("--no-committee", action="store_true",
                   help="Single balanced PM instead of a 3-PM committee.")
    # data / run
    p.add_argument("--n-tickers", type=int, default=0,
                   help="Universe cap (sets ARCHITECT_N_TICKERS); 0 = full.")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--run-id", default=None, help="Name for the output folder.")
    p.add_argument("--resume", action="store_true",
                   help="Continue from the run's existing DBs instead of fresh.")
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> BacktestConfig:
    personalities = (
        [PMPersonality.BALANCED] if args.no_committee else
        [PMPersonality.DEFENSIVE, PMPersonality.BALANCED, PMPersonality.AGGRESSIVE]
    )
    kwargs = dict(
        start=args.start, end=args.end, warmup=args.warmup, grid_freq=args.grid_freq,
        research_every=args.research_every, strategy_every=args.strategy_every,
        pm_rebalance_every=args.pm_every, run_research=not args.no_research,
        initial_strategies=args.initial_strategies,
        n_strategies_per_meeting=args.n_strategies,
        max_iterations=args.max_iterations, oos_ratio=args.oos_ratio,
        target_horizon=args.target_horizon, execution_model=args.execution,
        fund_max_positions=args.fund_max_positions,
        max_weight_per_name=args.max_weight_per_name,
        gross_leverage=args.gross_leverage, commission_bps=args.commission_bps,
        use_spread_cost=not args.no_spread_cost,
        committee=not args.no_committee, pm_personalities=personalities,
        pm_name="fund_committee" if not args.no_committee else "fund_pm",
        data_dir=args.data_dir,
        n_tickers=(args.n_tickers if args.n_tickers > 0 else None),
        fresh=not args.resume,
    )
    if args.run_id:
        kwargs["run_id"] = args.run_id
    return BacktestConfig(**kwargs)


def main() -> None:
    args = _parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")

    config = _config_from_args(args)
    log.info("Backtest run %s → %s", config.run_id, config.output_dir)
    results = run_backtest(config)
    print("\n" + "=" * 80)
    print(results.summary())
    print("=" * 80)
    print(f"\nArtifacts → {config.output_dir}")


if __name__ == "__main__":
    main()
