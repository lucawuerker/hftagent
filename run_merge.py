"""Compose an *active book* by merging selected scopes into ``data/books/<name>/``.

The seed-only main factor library is never touched.  This pools the researched
factors (and, by default, the fitted strategies + their artifacts/returns) of one
or more (config, prerun) scopes into a separate book the Portfolio Manager can
allocate across.

Examples
--------
::

    # Pool one prerun's factors + strategies into a book.
    python run_merge.py --from yfinance_equity_demo/gpt4omini120650 --into demo_book

    # Pool every prerun under a config.
    python run_merge.py --from-config yfinance_equity_demo --into demo_book

    # Selected factors only, no strategies, preview first.
    python run_merge.py --from yfinance_equity_demo/legacy_or_x --factors fac_a fac_b \
      --factors-only --dry-run

    # Pool + immediately run the PM committee over the composed book.
    python run_merge.py --from-config yfinance_equity_demo --into demo_book --run-pm
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from quant_fund_agent.merge import merge_into_book, parse_scope_ref
from quant_fund_agent.workspace import Book, Scope, list_scopes

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_merge")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from", dest="sources", nargs="+", metavar="CONFIG/PRERUN",
                     help="One or more source scopes (config/prerun).")
    src.add_argument("--from-config", default=None,
                     help="Pool every prerun under this config scope.")
    p.add_argument("--into", required=True, help="Destination book name (data/books/<name>).")
    p.add_argument("--factors", nargs="*", default=None,
                   help="Specific factor ids to merge (default: all researcher factors).")
    p.add_argument("--factors-only", action="store_true",
                   help="Merge factors only — do not pool strategies/artifacts.")
    p.add_argument("--on-collision", choices=["skip", "overwrite"], default="skip",
                   help="What to do when an id already exists in the book.")
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    p.add_argument("--run-pm", action="store_true",
                   help="After merging, run the PM committee over the composed book.")
    p.add_argument("--no-committee", action="store_true",
                   help="With --run-pm: single PM instead of a 3-PM committee.")
    return p.parse_args()


def _resolve_sources(args) -> list[Scope]:
    if args.from_config:
        scopes = [s for s in list_scopes() if s.config == args.from_config]
        if not scopes:
            raise SystemExit(f"No preruns found under config {args.from_config!r} "
                             f"in data/workspaces/.")
        return scopes
    return [parse_scope_ref(r) for r in args.sources]


def main() -> None:
    args = _parse_args()
    sources = _resolve_sources(args)
    book = Book(args.into)
    print(f"Merging {[s.label for s in sources]} → book {book.name!r}"
          f"{' (dry-run)' if args.dry_run else ''}")

    report = merge_into_book(
        sources, book,
        factor_ids=args.factors,
        include_strategies=not args.factors_only,
        on_collision=args.on_collision,
        dry_run=args.dry_run,
    )
    print("  " + report.summary())
    if report.promoted_factor_ids:
        print(f"  factors promoted : {report.promoted_factor_ids}")
    if report.promoted_strategy_ids:
        print(f"  strategies pooled: {report.promoted_strategy_ids}")
    for w in report.config_hash_warnings:
        log.warning(w)

    if args.run_pm and not args.dry_run:
        _run_pm_over_book(book, committee=not args.no_committee)


def _run_pm_over_book(book: Book, *, committee: bool) -> None:
    from quant_fund_agent import pipeline
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
    from quant_fund_agent.schemas import PMPersonality, VotingMethod

    sdb = StrategyDatabase(returns_dir=book.returns_dir)
    sdb.load_from_json(book.strategy_db_path)
    if not sdb.list_strategies():
        print("  PM skipped — the book has no strategies.")
        return
    pdb = PortfolioDatabase()
    if book.portfolio_db_path.exists():
        pdb.load_from_json(book.portfolio_db_path)

    record = pipeline.run_pm_rebalance(
        sdb, pdb,
        personalities=[PMPersonality.DEFENSIVE, PMPersonality.BALANCED,
                       PMPersonality.AGGRESSIVE],
        committee=committee,
        voting_method=VotingMethod.SIMPLE_AVERAGE,
        pm_name="book_committee" if committee else "book_pm",
    )
    book.ensure()
    pdb.save_to_json(book.portfolio_db_path)
    if record is not None:
        live = [a for a in record.allocations if a.enabled]
        print(f"  PM → {record.id}: {len(live)} live allocations "
              f"→ {book.portfolio_db_path}")


if __name__ == "__main__":
    main()
