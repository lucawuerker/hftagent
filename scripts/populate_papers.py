"""Populate data/papers/index.json with quantitative finance papers from ArXiv.

Searches ArXiv using a set of quant-finance query terms, generates a short
description for each paper via gpt-4o-mini, and upserts the results into the
paper database.  Already-indexed papers (matched by ArXiv ID stored in
metadata["arxiv_id"]) are skipped so the script is safe to re-run.

Usage:
    ./venv/bin/python scripts/populate_papers.py [--max-papers N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("populate_papers")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAPER_INDEX_PATH = Path(os.getenv("PAPER_INDEX_PATH", "data/papers/index.json"))

SEARCH_QUERIES = [
    # --- Limit order book & microstructure (data we have) ---------------
    "limit order book price prediction intraday",
    "order flow imbalance mid-price movement",
    "bid-ask spread dynamics liquidity intraday",
    "price impact market depth order book",
    "queue dynamics limit order book cancellation",
    "LOBSTER high frequency equity microstructure",

    # --- Price & volume momentum / reversal -----------------------------
    "cross-sectional momentum equity price returns",
    "time-series momentum trend following",
    "short-term price reversal mean reversion high frequency",
    "intraday momentum price continuation volume",
    "statistical arbitrage pairs trading price",

    # --- Volatility (computable from price/LOB) --------------------------
    "realized volatility forecasting intraday high frequency",
    "volatility clustering GARCH high frequency",
    "low volatility anomaly equity price",

    # --- Regime & market state ------------------------------------------
    "regime detection market state hidden Markov switching",
    "structural break market microstructure",

    # --- Advanced stochastic / mathematical finance ---------------------
    "path signature transform financial time series",
    "rough path rough volatility stochastic",
    "Hawkes process self-exciting order flow",
    "point process financial markets tick data",
]

LLM_MODEL = os.getenv("POPULATE_PAPERS_LLM_MODEL", "gpt-4o-mini")
ARXIV_MAX_RESULTS_PER_QUERY = 30  # fetched per query before dedup / cap
# Extended to include math/stat categories where rough-path, Hawkes, and
# signature-transform papers are typically filed.
ARXIV_CATEGORIES = [
    "q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.MF", "q-fin.CP",
    "econ.GN", "cs.LG", "stat.ML", "math.PR",
]


# ---------------------------------------------------------------------------
# ArXiv search
# ---------------------------------------------------------------------------

def _fetch_arxiv(query: str, max_results: int) -> list[dict]:
    import arxiv

    cat_filter = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    full_query = f"({query}) AND ({cat_filter})"

    client = arxiv.Client(page_size=min(max_results, 100), delay_seconds=1.0)
    search = arxiv.Search(
        query=full_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = []
    for r in client.results(search):
        results.append({
            "arxiv_id": r.get_short_id(),
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "abstract": r.summary.strip(),
            "published_date": r.published.date() if r.published else None,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
        })
    return results


# ---------------------------------------------------------------------------
# Description generation
# ---------------------------------------------------------------------------

def _generate_description(title: str, abstract: str, llm) -> str:
    prompt = (
        "You are a quantitative researcher. Given the paper title and abstract below, "
        "write exactly 2-3 sentences describing: (1) the core trading signal or methodology, "
        "(2) why it might generate alpha. Be specific and concise. No fluff.\n\n"
        f"Title: {title}\n\nAbstract: {abstract[:1500]}"
    )
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        log.warning("description generation failed for '%s': %s", title[:60], e)
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:60] or "paper"


def _load_existing_arxiv_ids(paper_db) -> set[str]:
    ids = set()
    for p in paper_db.list_papers():
        aid = p.metadata.get("arxiv_id")
        if aid:
            ids.add(aid)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate paper database from ArXiv")
    parser.add_argument("--max-papers", type=int, default=300,
                        help="Maximum number of new papers to add (default: 300)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing anything")
    args = parser.parse_args()

    # -- Load existing DB ------------------------------------------------
    from quant_fund_agent.databases import PaperDatabase
    from quant_fund_agent.schemas import Paper, PaperStatus

    paper_db = PaperDatabase()
    paper_db.load_from_json(PAPER_INDEX_PATH)
    existing_arxiv_ids = _load_existing_arxiv_ids(paper_db)
    existing_ids = {p.id for p in paper_db.list_papers()}
    log.info("Existing papers in DB: %d", len(existing_ids))

    # -- Collect candidates from ArXiv -----------------------------------
    seen_arxiv: set[str] = set(existing_arxiv_ids)
    candidates: list[dict] = []

    for query in SEARCH_QUERIES:
        if len(candidates) >= args.max_papers * 3:
            break
        log.info("Searching: %s", query)
        try:
            results = _fetch_arxiv(query, max_results=ARXIV_MAX_RESULTS_PER_QUERY)
        except Exception as e:
            log.warning("ArXiv query failed ('%s'): %s", query, e)
            continue
        for r in results:
            if r["arxiv_id"] not in seen_arxiv:
                seen_arxiv.add(r["arxiv_id"])
                candidates.append(r)
        log.info("  → %d unique new candidates so far", len(candidates))
        time.sleep(3.0)  # avoid ArXiv 429s between queries

    candidates = candidates[: args.max_papers]
    log.info("Total new candidates to process: %d", len(candidates))

    if args.dry_run:
        for c in candidates[:20]:
            print(f"  [{c['arxiv_id']}] {c['title'][:80]}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
        return

    # -- Set up LLM ------------------------------------------------------
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

    # -- Generate descriptions and build Paper objects -------------------
    new_papers: list[Paper] = []
    for i, c in enumerate(candidates, 1):
        log.info("[%d/%d] %s", i, len(candidates), c["title"][:70])

        description = _generate_description(c["title"], c["abstract"], llm)

        base_id = _slugify(c["title"])
        paper_id = base_id
        suffix = 1
        while paper_id in existing_ids:
            suffix += 1
            paper_id = f"{base_id}_{suffix}"
        existing_ids.add(paper_id)

        paper = Paper(
            id=paper_id,
            title=c["title"],
            authors=c["authors"],
            abstract=c["abstract"],
            published_date=c["published_date"],
            status=PaperStatus.UNREAD,
            file_path=None,
            url=c["url"],
            metadata={
                "arxiv_id": c["arxiv_id"],
                "pdf_url": c["pdf_url"],
                "description": description,
            },
        )
        paper_db.add_paper(paper)
        new_papers.append(paper)

        # Save incrementally every 25 papers so progress isn't lost on interrupt
        if i % 25 == 0:
            paper_db.save_to_json(PAPER_INDEX_PATH)
            log.info("  Checkpoint saved (%d papers written so far)", i)

    paper_db.save_to_json(PAPER_INDEX_PATH)
    log.info("Done. Added %d new papers. Total in DB: %d",
             len(new_papers), len(paper_db.list_papers()))


if __name__ == "__main__":
    main()
