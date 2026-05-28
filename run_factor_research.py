"""Run a single Factor Researcher Agent session.

Usage::

    export OPENAI_API_KEY=sk-...
    # default session: 2 papers, 3 ideas, |IC@6| >= 0.01
    python run_factor_research.py

    # weekly-style session with a cutoff and a session id
    python run_factor_research.py \
        --session-id 2019-W03 \
        --cutoff 2019-01-21 \
        --n-papers 2 \
        --n-ideas 3 \
        --ic-threshold 0.01 \
        --horizon 6

    # nuke previous researcher artefacts (code files + DB entries) first
    python run_factor_research.py --reset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=date.today().isoformat(),
                        help="Tag used to mark researcher factors from this run.")
    parser.add_argument("--cutoff", default=None,
                        help="Lookahead-bias cutoff date YYYY-MM-DD. "
                             "Only papers and bar data strictly before this "
                             "date are visible to the agent.")
    parser.add_argument("--n-papers", type=int, default=2,
                        help="How many papers to read this session.")
    parser.add_argument("--n-ideas", type=int, default=3,
                        help="How many factor ideas the LLM should propose.")
    parser.add_argument("--ic-threshold", type=float, default=0.01,
                        help="Minimum absolute IC at the target horizon to "
                             "keep a factor.")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Forward-return horizon in 10-sec bars used as "
                             "the IC gate (default 6 = 1 minute).")
    parser.add_argument("--sample-strategy", choices=("unread_first", "random"),
                        default="unread_first")
    parser.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"),
                        help="Path to the ticker_data root (panel source).")
    parser.add_argument("--n-tickers", type=int, default=15,
                        help="Cap the universe loaded for the research-time "
                             "IC backtest (default 15).  Pass 0 to load every "
                             "ticker in --data-dir (memory-heavy).")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe ALL prior researcher factors (code + DB) "
                             "before running.")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY in .env or environment.",
              file=sys.stderr)
        sys.exit(1)

    from quant_fund_agent.agents.factor_research.graph import (
        factor_research_graph,
        reset_researcher_state,
    )
    from quant_fund_agent.agents.factor_research.state import FactorResearcherState
    from quant_fund_agent.factors import discover_factors

    discover_factors()  # seed factors

    if args.reset:
        print("=" * 80)
        print("RESET: purging all previously-generated researcher factors")
        print("=" * 80)
        reset_researcher_state()

    print("\n" + "=" * 80)
    print(f"FACTOR RESEARCHER AGENT  (session={args.session_id})")
    print(f"  papers={args.n_papers}  ideas={args.n_ideas}  "
          f"|IC@{args.horizon}| ≥ {args.ic_threshold}  "
          f"cutoff={args.cutoff or '(none)'}  "
          f"universe={args.n_tickers if args.n_tickers > 0 else 'ALL'}")
    print("=" * 80)

    init = FactorResearcherState(
        session_id=args.session_id,
        cutoff_date=_parse_date(args.cutoff),
        n_papers=args.n_papers,
        n_factor_ideas=args.n_ideas,
        ic_threshold=args.ic_threshold,
        ic_target_horizon=args.horizon,
        paper_sample_strategy=args.sample_strategy,
        data_dir=args.data_dir,
        n_tickers=(args.n_tickers if args.n_tickers > 0 else None),
    )

    result = factor_research_graph.invoke(init)

    # LangGraph returns the final state as a dict.
    selected_papers = result.get("selected_papers", [])
    factor_ideas = result.get("factor_ideas", [])
    candidates = result.get("candidates", [])
    kept = result.get("kept_factor_ids", [])
    rejected = result.get("rejected_factor_ids", [])

    print("\n" + "-" * 80)
    print(f"PAPERS READ ({len(selected_papers)})")
    print("-" * 80)
    for s in selected_papers:
        title = s.title if hasattr(s, "title") else s.get("title", "")
        pid = s.paper_id if hasattr(s, "paper_id") else s.get("paper_id", "")
        n_chars = len(s.text) if hasattr(s, "text") else len(s.get("text", ""))
        print(f"  - [{pid}] {title}  ({n_chars} chars extracted)")

    print("\n" + "-" * 80)
    print(f"BRAINSTORMED IDEAS ({len(factor_ideas)})")
    print("-" * 80)
    for i in factor_ideas:
        fid = i.factor_id if hasattr(i, "factor_id") else i.get("factor_id", "")
        name = i.name if hasattr(i, "name") else i.get("name", "")
        cat = i.category if hasattr(i, "category") else i.get("category", "")
        idea = (i.trading_idea if hasattr(i, "trading_idea")
                else i.get("trading_idea", ""))
        print(f"  - {fid} ({cat}): {name}")
        print(f"      {idea[:160]}")

    print("\n" + "-" * 80)
    print(f"CANDIDATES AFTER BACKTEST ({len(candidates)})")
    print("-" * 80)
    for c in candidates:
        fid = c.idea.factor_id if hasattr(c, "idea") else c.get("idea", {}).get("factor_id")
        bm = c.backtest_metrics if hasattr(c, "backtest_metrics") else c.get("backtest_metrics")
        reject = c.rejected_reason if hasattr(c, "rejected_reason") else c.get("rejected_reason", "")
        ic = getattr(bm, "information_coefficient", None) if bm else None
        icir = getattr(bm, "ic_ir", None) if bm else None
        print(f"  - {fid}: IC={ic}  ICIR={icir}  reject={reject!r}")

    print("\n" + "-" * 80)
    print(f"OUTCOME")
    print("-" * 80)
    print(f"  KEPT     ({len(kept)}): {kept}")
    print(f"  REJECTED ({len(rejected)}): {rejected}")


if __name__ == "__main__":
    main()
