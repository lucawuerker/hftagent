"""Showcase — Stage 1: Factor Researcher Agent (no market data required).

Runs the Factor Researcher exactly as the real pipeline does, up to the point
where it would need the order-book / price panel:

    load_papers  →  brainstorm  →  generate_code   (this script)
    ───────────────────────────────────────────────────────────────
    backtest_factors  →  filter_and_persist        (SKIPPED — needs ticker_data/)

So the agent reads real research papers, brainstorms alpha ideas and writes a
complete ``BaseFactor`` subclass for each idea (validated + smoke-tested on
synthetic data inside ``codegen.materialise``).  The IC backtest that would
normally decide which factors to keep is skipped, so every generated factor is
persisted as a CANDIDATE (generated, not yet validated) into a *separate
showcase factor database*.

What your supervisor can inspect afterwards:
  * the generated subclass files in ``quant_fund_agent/factors/researcher/``
  * the new entries in ``data/factors/showcase_factor_db.json``

Usage::

    ./venv/bin/python showcase_pipeline/1_factor_research.py
    ./venv/bin/python showcase_pipeline/1_factor_research.py --n-papers 2 --n-ideas 3
    ./venv/bin/python showcase_pipeline/1_factor_research.py --reset   # fresh showcase DB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import showcase_common as sc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--session-id", default="showcase",
                   help="Tag stored on each generated factor.")
    p.add_argument("--n-papers", type=int, default=2,
                   help="How many research papers to read this session.")
    p.add_argument("--n-ideas", type=int, default=3,
                   help="How many factor ideas the LLM should brainstorm.")
    p.add_argument("--sample", choices=("random", "unread_first"), default="random",
                   help="How to pick papers from the full index.json "
                        "(default: random, to surface the breadth of the index).")
    p.add_argument("--reset", action="store_true",
                   help="Rebuild the showcase factor DB from the real DB first.")
    p.add_argument("--no-code", action="store_true",
                   help="Do not print the full generated source of each factor.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sc.banner("SHOWCASE · STAGE 1 · FACTOR RESEARCHER")
    print("Reads papers → brainstorms ideas → writes BaseFactor subclasses.")
    print("The IC backtest that validates them is SKIPPED (no market data here),")
    print("so every generated factor is stored as an unvalidated CANDIDATE.")

    sc.require_openai_key()
    sc.ensure_showcase_factor_db(reset=args.reset)

    # Imports happen after env setup so the showcase factor DB / in-process
    # tooling are picked up correctly.
    from quant_fund_agent.agents.factor_research.graph import (
        _coerce_category,
        brainstorm,
        generate_code,
        load_papers,
    )
    from quant_fund_agent.agents.factor_research.state import FactorResearcherState
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.schemas import FactorRecord, FactorSource, FactorStatus

    discover_factors()  # register the seed factors (so brainstorm avoids ID clashes)

    state = FactorResearcherState(
        session_id=args.session_id,
        n_papers=args.n_papers,
        n_factor_ideas=args.n_ideas,
        paper_sample_strategy=args.sample,
    )

    # ── load_papers → brainstorm → generate_code ─────────────────────────
    state = state.model_copy(update=load_papers(state))
    state = state.model_copy(update=brainstorm(state))
    state = state.model_copy(update=generate_code(state))

    # ── Report: papers read ──────────────────────────────────────────────
    _source_label = {
        "local_pdf": "local PDF",
        "link": "full text fetched via paper link",
        "abstract": "abstract only (no PDF / link unavailable)",
    }
    sc.section(f"PAPERS READ ({len(state.selected_papers)})")
    for s in state.selected_papers:
        src = _source_label.get(s.source, s.source or "unknown")
        print(f"  - [{s.paper_id}] {s.title}")
        print(f"      {len(s.text or ''):>6} chars extracted  ·  source: {src}")

    # ── Report: brainstormed ideas ───────────────────────────────────────
    sc.section(f"BRAINSTORMED IDEAS ({len(state.factor_ideas)})")
    for idea in state.factor_ideas:
        print(f"  - {idea.factor_id} ({idea.category}): {idea.name}")
        print(f"      {idea.trading_idea[:200]}")

    # ── Persist every materialised candidate into the showcase factor DB ─
    db = FactorDatabase()
    db.load_from_json(sc.FACTOR_DB_SHOWCASE)

    added: list[FactorRecord] = []
    failed: list[tuple[str, str]] = []
    for cand in state.candidates:
        fid = cand.idea.factor_id
        if not cand.code_path:  # codegen failed for this idea
            failed.append((fid, cand.rejected_reason or "codegen failed"))
            continue
        record = FactorRecord(
            id=fid,
            name=cand.idea.name,
            description=cand.idea.description,
            trading_idea=cand.idea.trading_idea,
            category=_coerce_category(cand.idea.category),
            status=FactorStatus.CANDIDATE,   # generated, not yet IC-validated
            backtest_metrics=None,
            source_paper_ids=cand.idea.source_paper_ids,
            source=FactorSource.RESEARCHER,
            research_session_id=state.session_id,
            code_path=cand.code_path,
            metadata={
                "showcase": True,
                "note": "Generated by the Factor Researcher; IC backtest "
                        "skipped (no market data in this repository).",
            },
        )
        db.add_factor(record)
        added.append(record)

    db.save_to_json(sc.FACTOR_DB_SHOWCASE)

    # ── Report: where each factor was added ──────────────────────────────
    db_file = sc.FACTOR_DB_SHOWCASE
    sc.section(f"GENERATED FACTORS ADDED ({len(added)})")
    for rec in added:
        cp = Path(rec.code_path)
        print(f"\n  ✓ {rec.id}  —  {rec.name}  [{rec.category.value}]")
        print(f"      SAVED AS subclass file : {cp.name}")
        print(f"               in folder     : {cp.parent}/")
        print(f"      ADDED to factor DB file: {db_file.name}  "
              f"(in {db_file.parent}/, status={rec.status.value})")
        if not args.no_code:
            try:
                code = open(rec.code_path, encoding="utf-8").read()
            except OSError as e:
                code = f"(could not read source: {e})"
            print("      ┄┄┄┄┄ generated source ┄┄┄┄┄")
            for line in code.splitlines():
                print(f"      │ {line}")
            print("      ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")

    if failed:
        sc.section(f"CODEGEN FAILURES ({len(failed)})")
        for fid, reason in failed:
            print(f"  ✗ {fid}: {reason}")

    sc.note_skipped_data_step(
        "the single-factor IC backtest that decides which factors to keep "
        "(backtest_factors → filter_and_persist)."
    )

    sc.banner("STAGE 1 COMPLETE")
    if added:
        print(f"  {len(added)} factor subclass file(s) written under "
              f"{sc.RESEARCHER_FACTOR_DIR}/ :")
        for rec in added:
            print(f"      - {Path(rec.code_path).name}")
        print(f"  ...and added to factor DB file: {sc.FACTOR_DB_SHOWCASE.name} "
              f"(in {sc.FACTOR_DB_SHOWCASE.parent}/)")
    else:
        print("  No factors were generated this run.")
    print("  Next: ./venv/bin/python showcase_pipeline/2_selector.py")


if __name__ == "__main__":
    main()
