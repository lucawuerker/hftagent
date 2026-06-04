"""Stage 1: Factor Researcher Agent — full run on the local market data.

Runs the Factor Researcher end-to-end, including the single-factor IC backtest
that the showcase version skips:

    load_papers → brainstorm → generate_code → backtest_factors → filter_and_persist

So the agent reads real research papers, brainstorms alpha ideas, writes a
``BaseFactor`` subclass per idea, **backtests each one against the local panel**,
and persists only the factors whose |IC| at the target horizon clears the
threshold into the **real** factor database (``data/factors/factor_db.json``).

Usage::

    ./venv/bin/python run_pipeline/1_factor_research.py
    ./venv/bin/python run_pipeline/1_factor_research.py --n-papers 2 --n-ideas 3 --n-tickers 8
    ./venv/bin/python run_pipeline/1_factor_research.py --session-id 2019-W03 --cutoff 2019-01-21
    ./venv/bin/python run_pipeline/1_factor_research.py --reset   # purge prior researcher factors
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

import pipeline_common as pc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_date(s: str | None) -> date | None:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--session-id", default=date.today().isoformat(),
                   help="Tag stored on each factor generated this session.")
    p.add_argument("--cutoff", default=None,
                   help="Lookahead-bias cutoff YYYY-MM-DD: only papers and bar "
                        "data strictly before this date are visible.")
    p.add_argument("--n-papers", type=int, default=2,
                   help="How many research papers to read this session.")
    p.add_argument("--n-ideas", type=int, default=3,
                   help="How many factor ideas the LLM should brainstorm.")
    p.add_argument("--horizon", type=int, default=6,
                   help="Horizon in 10-sec bars at which each factor's IC is "
                        "measured and recorded (default 6 = 1 minute).  IC is "
                        "recorded for reference only — it is not a keep/drop gate.")
    p.add_argument("--sample-strategy", choices=("unread_first", "random"),
                   default="unread_first",
                   help="How to pick papers from the full index.json.")
    p.add_argument("--reset", action="store_true",
                   help="Purge ALL prior researcher factors (code + DB) first.")
    pc.add_common_args(p)  # --mcp, --n-tickers (default 15, 0=all), --data-dir
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.banner(f"STAGE 1 · FACTOR RESEARCHER  (session={args.session_id})")
    print("Reads papers → brainstorms ideas → writes BaseFactor subclasses →")
    print("backtests each factor on the local panel → keeps every one that runs")
    print("(IC is recorded for reference, not used as a keep/drop gate).")

    pc.require_openai_key()
    pc.require_market_data(args.data_dir)
    pc.configure_runtime(use_mcp=args.mcp)  # research uses state.data_dir/n_tickers

    from quant_fund_agent.agents.factor_research.graph import (
        factor_research_graph,
        reset_researcher_state,
    )
    from quant_fund_agent.agents.factor_research.state import FactorResearcherState
    from quant_fund_agent.factors import discover_factors

    discover_factors()  # register seed factors (so brainstorm avoids ID clashes)

    if args.reset:
        pc.section("RESET: purging all previously-generated researcher factors")
        reset_researcher_state()

    n_tickers = args.n_tickers if args.n_tickers > 0 else None
    print(f"\n  papers={args.n_papers}  ideas={args.n_ideas}  "
          f"IC@{args.horizon} recorded (not gated)  "
          f"cutoff={args.cutoff or '(none)'}  "
          f"universe={n_tickers if n_tickers else 'ALL'}  "
          f"data_dir={args.data_dir}")

    state = FactorResearcherState(
        session_id=args.session_id,
        cutoff_date=_parse_date(args.cutoff),
        n_papers=args.n_papers,
        n_factor_ideas=args.n_ideas,
        ic_target_horizon=args.horizon,
        paper_sample_strategy=args.sample_strategy,
        data_dir=args.data_dir,
        n_tickers=n_tickers,
    )

    result = factor_research_graph.invoke(state)

    # LangGraph returns the final state as a dict.
    selected_papers = result.get("selected_papers", [])
    factor_ideas = result.get("factor_ideas", [])
    candidates = result.get("candidates", [])
    kept = result.get("kept_factor_ids", [])
    rejected = result.get("rejected_factor_ids", [])

    _source_label = {
        "local_pdf": "local PDF",
        "link": "full text fetched via paper link",
        "abstract": "abstract only (no PDF / link unavailable)",
    }
    pc.section(f"PAPERS READ ({len(selected_papers)})")
    for s in selected_papers:
        src = _source_label.get(s.source, s.source or "unknown")
        print(f"  - [{s.paper_id}] {s.title}")
        print(f"      {len(s.text or ''):>6} chars extracted  ·  source: {src}")

    pc.section(f"BRAINSTORMED IDEAS ({len(factor_ideas)})")
    for idea in factor_ideas:
        print(f"  - {idea.factor_id} ({idea.category}): {idea.name}")
        print(f"      {idea.trading_idea[:200]}")

    pc.section(f"CANDIDATES AFTER IC BACKTEST ({len(candidates)})")
    for c in candidates:
        bm = c.backtest_metrics
        ic = getattr(bm, "information_coefficient", None) if bm else None
        icir = getattr(bm, "ic_ir", None) if bm else None
        reject = c.rejected_reason or ""
        print(f"  - {c.idea.factor_id}: IC={pc.fmt(ic, 4)}  ICIR={pc.fmt(icir, 4)}"
              + (f"  reject={reject!r}" if reject else ""))

    pc.section("OUTCOME")
    print(f"  KEPT     ({len(kept)}): {kept}")
    print(f"  REJECTED ({len(rejected)}): {rejected}")

    pc.banner("STAGE 1 COMPLETE")
    print("  Every factor that ran was persisted to the real factor DB "
          "(status=backtested);")
    print("  only codegen/backtest failures were dropped.")
    print("  Next: ./venv/bin/python run_pipeline/2_selector.py")


if __name__ == "__main__":
    main()
