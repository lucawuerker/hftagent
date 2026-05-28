"""Run the Portfolio Manager Agent over the current StrategyDatabase.

Examples
--------
::

    # SELECTOR mode, balanced personality, default construction (risk parity).
    python run_portfolio_manager.py

    # SELECTOR mode but force equal-weight, defensive PM, 12 strategies.
    python run_portfolio_manager.py \
        --personality defensive --method equal_weight --target-n 12

    # ACTIVE mode, aggressive PM — the LLM picks the construction method.
    python run_portfolio_manager.py --mode active --personality aggressive

    # Run BOTH a defensive and an aggressive PM side-by-side over the same DB.
    python run_portfolio_manager.py --personality defensive
    python run_portfolio_manager.py --personality aggressive

    # Committee: 3 PMs propose, simple average aggregates them into one
    # consensus PortfolioRecord (no LLM call).
    python run_portfolio_manager.py --committee defensive,balanced,aggressive

    # Committee with an LLM moderator that reads all proposals and
    # synthesises the consensus allocation.
    python run_portfolio_manager.py --committee defensive,balanced,aggressive \
        --committee-voting llm_moderator

The script:
  1. Loads the persisted StrategyDatabase (if any) from
     ``data/strategies/strategy_db.json``.
  2. Refreshes the correlation matrix (so existing strategies — which
     pre-date the PM layer — get their correlations populated).
  3. Invokes the PM agent.
  4. Prints the resulting allocation and persists the PortfolioDatabase
     to ``data/portfolio/portfolio_db.json``.

If the strategy DB is empty the script bails out with a helpful message.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-18s  %(message)s",
    datefmt="%H:%M:%S",
)

STRATEGY_DB_PATH = Path("data/strategies/strategy_db.json")
PORTFOLIO_DB_PATH = Path("data/portfolio/portfolio_db.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pm-name", default="default_pm",
                   help="Name for this PM (used in the audit log).")
    p.add_argument("--mode", choices=["selector", "active"], default="selector",
                   help="selector = passive (fixed construction); "
                        "active = LLM picks the construction method too.")
    p.add_argument("--personality",
                   choices=["defensive", "balanced", "aggressive"],
                   default="balanced")
    p.add_argument(
        "--method",
        choices=[
            "equal_weight", "inverse_volatility", "min_variance",
            "mean_variance", "max_sharpe", "risk_parity",
            "hierarchical_risk_parity",
        ],
        default=None,
        help="Override the construction method (SELECTOR mode only). "
             "If omitted, uses the personality's default.",
    )
    p.add_argument("--target-n", type=int, default=None,
                   help="Override the personality's target_n_strategies.")
    p.add_argument("--annualisation-factor", type=float, default=1.0,
                   help="Multiplier applied to per-bar covariance "
                        "(e.g. bars_per_year for intraday).")
    p.add_argument(
        "--committee", default=None,
        help="Comma-separated list of personalities to run as a "
             "committee, e.g. 'defensive,balanced,aggressive'.  When "
             "set, --personality is ignored and each listed PM is run "
             "in propose-only mode; the committee aggregates them into "
             "one consensus PortfolioRecord.",
    )
    p.add_argument(
        "--committee-voting",
        choices=["simple_average", "weighted_average", "llm_moderator"],
        default="simple_average",
        help="How the committee aggregates per-PM proposals.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Imports ──────────────────────────────────────────────────────
    from quant_fund_agent.agents.portfolio_manager.graph import (
        portfolio_manager_graph,
    )
    from quant_fund_agent.agents.portfolio_manager.state import (
        PortfolioManagerState,
    )
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
    from quant_fund_agent.portfolio import get_personality_profile
    from quant_fund_agent.schemas import (
        ConstructionMethod,
        PMMode,
        PMPersonality,
        VotingMethod,
    )
    from quant_fund_agent.agents.portfolio_manager.committee import (
        CommitteeConfig,
        run_pm_committee,
    )

    # ── Load DBs ─────────────────────────────────────────────────────
    strategy_db = StrategyDatabase()
    strategy_db.load_from_json(STRATEGY_DB_PATH)

    portfolio_db = PortfolioDatabase()
    portfolio_db.load_from_json(PORTFOLIO_DB_PATH)

    n_strategies = len(strategy_db.list_strategies())
    if n_strategies == 0:
        print(
            f"No strategies found at {STRATEGY_DB_PATH}.\n"
            "Run the architect → statistician pipeline first to populate it."
        )
        return

    print(f"\nLoaded {n_strategies} strategies from {STRATEGY_DB_PATH}.\n")

    # Backfill correlations for any pre-existing strategies (the user
    # explicitly asked for this).  ``load_from_json`` already does this,
    # but we call it again here for visibility.
    corr = strategy_db.refresh_correlations()
    if not corr.empty:
        print(f"Refreshed correlation matrix: {corr.shape[0]} × {corr.shape[1]}")
    else:
        print(
            "Warning: correlation matrix is empty — no persisted returns "
            "files were found for any strategy.  The PM will fall back to "
            "vol-only covariance estimates."
        )

    # ── Build PM state(s) ────────────────────────────────────────────
    method_override = ConstructionMethod(args.method) if args.method else None

    if args.committee:
        # Committee path: build one PM per listed personality.
        personalities = [
            PMPersonality(p.strip())
            for p in args.committee.split(",")
            if p.strip()
        ]
        if len(personalities) < 2:
            raise SystemExit(
                "--committee needs at least two personalities, e.g. "
                "'defensive,aggressive'.  For a single PM, drop --committee."
            )
        pm_states = [
            PortfolioManagerState(
                pm_name=f"{args.pm_name}_{p.value}",
                mode=PMMode(args.mode),
                personality=p,
                profile=get_personality_profile(p),
                construction_method_override=method_override,
                target_n_strategies=args.target_n,
                annualisation_factor=args.annualisation_factor,
                strategy_db=strategy_db,
                portfolio_db=portfolio_db,
            )
            for p in personalities
        ]
        cfg = CommitteeConfig(
            voting_method=VotingMethod(args.committee_voting),
            pm_name=args.pm_name,
        )

        print("=" * 80)
        print(f"Running COMMITTEE '{args.pm_name}' over PMs: "
              f"{[p.value for p in personalities]}  "
              f"(voting={args.committee_voting})")
        print("=" * 80)
        record = run_pm_committee(pm_states, cfg, strategy_db, portfolio_db)
    else:
        # Single-PM path (unchanged behaviour).
        personality = PMPersonality(args.personality)
        profile = get_personality_profile(personality)
        state = PortfolioManagerState(
            pm_name=args.pm_name,
            mode=PMMode(args.mode),
            personality=personality,
            profile=profile,
            construction_method_override=method_override,
            target_n_strategies=args.target_n,
            annualisation_factor=args.annualisation_factor,
            strategy_db=strategy_db,
            portfolio_db=portfolio_db,
        )
        print("=" * 80)
        print(f"Running PM '{args.pm_name}' "
              f"(mode={args.mode}, personality={args.personality})")
        print("=" * 80)
        result = portfolio_manager_graph.invoke(state)
        record = result.get("portfolio_record")

    if record is None:
        print("\nPM produced no portfolio record.")
        return

    # ── Print results ────────────────────────────────────────────────
    print("\nPORTFOLIO ALLOCATION")
    print("-" * 80)
    print(f"Portfolio ID:        {record.id}")
    print(f"PM name:             {record.pm_name}")
    print(f"Mode / personality:  {record.mode.value} / {record.personality.value}")
    print(f"Construction method: {record.construction_method.value}")
    print(f"# strategies:        {len(record.allocations)}")

    print("\nAllocations:")
    for a in record.allocations:
        if not a.enabled:
            continue
        print(f"  - {a.strategy_id:<35s}  w = {a.weight:+.4f}")

    if record.flagged_strategy_ids:
        print(f"\nFlagged for review: {record.flagged_strategy_ids}")
    if record.retired_strategy_ids:
        print(f"Retired:            {record.retired_strategy_ids}")

    if record.expected_metrics:
        print("\nExpected portfolio metrics:")
        for k, v in record.expected_metrics.items():
            print(f"  {k}: {v}")

    if record.pm_rationale:
        print("\nRationale:")
        print(record.pm_rationale)

    # ── Persist DBs ──────────────────────────────────────────────────
    portfolio_db.save_to_json(PORTFOLIO_DB_PATH)
    strategy_db.save_to_json(STRATEGY_DB_PATH)
    print(f"\nPortfolio DB written to {PORTFOLIO_DB_PATH}")
    print(f"Strategy DB updated at {STRATEGY_DB_PATH} (pm_status fields refreshed)")


if __name__ == "__main__":
    main()
