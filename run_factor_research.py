"""One-off **permanent** factor-research prerun.

Mine a large ensemble of alpha factors from the paper library *once* and keep
them **permanently** in the global factor DB (``data/factors/factor_db.json``) +
the package ``factors/researcher/`` code dir.  This is the library a subsequent
walk-forward backtest draws on when run with ``--factor-universe all``.

It differs from the research meetings *inside* a backtest in two ways:

- **Persistence is permanent** — it writes the global factor DB (a backtest writes
  a run-scoped DB and never touches the global one).
- **Paper read-tracking is its own scenario** — it keeps a permanent read-log at
  ``data/papers/prerun_read_log.json`` so successive sessions advance through the
  714-paper library without re-reading, and without biasing any backtest's paper
  selection (each backtest has its own run-scoped read-log).

The session loop keeps researching (broad paper sample, ~1 idea per paper) until
``--target-factors`` researcher factors exist in the global DB or ``--max-sessions``
is reached.

Examples
--------
::

    # The real prerun: ~100 factors, broad sample of the library.
    python run_factor_research.py --target-factors 100

    # Smoke run, in-process, tiny.
    QF_USE_MCP=0 python run_factor_research.py \
      --target-factors 4 --papers-per-session 4 --ideas-per-session 4 \
      --max-sessions 2 --n-tickers 8

    # Start the permanent researcher library over from scratch.
    python run_factor_research.py --reset --target-factors 100
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_factor_research")

# Permanent, prerun-scoped "papers already read" log (kept across preruns so the
# library is covered without re-reading; cleared by --reset).
PRERUN_READ_LOG = Path("data/papers/prerun_read_log.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target-factors", type=int, default=100,
                   help="Stop once the global DB holds this many researcher factors.")
    p.add_argument("--papers-per-session", type=int, default=25)
    p.add_argument("--ideas-per-session", type=int, default=25,
                   help="Total idea budget per session (split across the papers).")
    p.add_argument("--max-sessions", type=int, default=12,
                   help="Hard cap on the number of research sessions.")
    p.add_argument("--horizon", type=int, default=6,
                   help="Bars at which each factor's IC is recorded (not gated).")
    p.add_argument("--n-tickers", type=int, default=15,
                   help="Universe cap for the research-time IC backtest.")
    p.add_argument("--sample", choices=["random", "unread_first"], default="random",
                   help="How to pick each session's papers from the unread pool.")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--reset", action="store_true",
                   help="Purge the existing permanent researcher factors + code "
                        "and the prerun read-log before starting.")
    return p.parse_args()


def _researcher_factor_count() -> int:
    """How many RESEARCHER factors are currently in the GLOBAL factor DB."""
    from quant_fund_agent import pipeline
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    db = FactorDatabase()
    db.load_from_json(pipeline.FACTOR_DB_PATH)
    return len(db.list_factors(source=FactorSource.RESEARCHER))


def main() -> None:
    args = _parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")

    # Permanent persistence: leave FACTOR_DB_PATH at its default (the global DB).
    os.environ.pop("FACTOR_DB_PATH", None)
    os.environ["DATA_DIR"] = args.data_dir
    # Prerun-scoped paper read-log (its own scenario; never the global index).
    os.environ["PAPER_READ_LOG"] = str(PRERUN_READ_LOG)

    from quant_fund_agent import pipeline
    from quant_fund_agent.agents.factor_research.graph import reset_researcher_state

    if args.reset:
        log.info("Resetting permanent researcher state (factors, code, read-log) …")
        reset_researcher_state()
        PRERUN_READ_LOG.unlink(missing_ok=True)

    start_count = _researcher_factor_count()
    log.info("Prerun start: %d researcher factor(s) in %s; target %d",
             start_count, pipeline.FACTOR_DB_PATH, args.target_factors)

    for i in range(1, args.max_sessions + 1):
        have = _researcher_factor_count()
        if have >= args.target_factors:
            log.info("Target reached (%d ≥ %d) — stopping.", have, args.target_factors)
            break
        log.info("── prerun session %d/%d  (have %d, want %d) ──",
                 i, args.max_sessions, have, args.target_factors)
        try:
            rs = pipeline.run_research_session(
                session_id=f"prerun-{i:03d}",
                cutoff_date=None,                       # whole library (per decision)
                n_papers=args.papers_per_session,
                n_ideas=args.ideas_per_session,
                horizon=args.horizon,
                n_tickers=args.n_tickers,
                paper_sample_strategy=args.sample,
                data_dir=args.data_dir,
            )
            kept = rs.get("kept_factor_ids", [])
            log.info("session %d kept %d factor(s): %s", i, len(kept), kept)
            if not rs.get("selected_papers"):
                log.warning("No papers left to read — stopping early.")
                break
        except Exception as e:  # noqa: BLE001 — one bad session must not abort the prerun
            log.warning("prerun session %d failed: %s", i, e)

    _summarise(pipeline)


def _summarise(pipeline) -> None:
    from collections import Counter

    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    db = FactorDatabase()
    db.load_from_json(pipeline.FACTOR_DB_PATH)
    researcher = db.list_factors(source=FactorSource.RESEARCHER)
    by_cat = Counter(f.category.value for f in researcher)

    print("\n" + "=" * 80)
    print(f"Permanent factor library: {pipeline.FACTOR_DB_PATH}")
    print(f"  seed factors      : {len(db.list_factors(source=FactorSource.SEED))}")
    print(f"  researcher factors: {len(researcher)}")
    print(f"  by category       : {dict(by_cat)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
