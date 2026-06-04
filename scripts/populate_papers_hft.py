"""Populate data/papers/index.json with HFT / microstructure papers from ArXiv.

A more specialised companion to ``populate_papers.py``: searches ArXiv with a
set of ultra-high-frequency, market-microstructure, rough-path and point-process
query terms, generates a short alpha-oriented description for each paper via
gpt-4o-mini, and upserts the results into the existing paper database.

Deduplication is enforced on the ArXiv ID: candidates already in the DB (or seen
earlier in this run) are skipped, so **no paper is ever added more than once**
and the script is safe to re-run.  Records are written in the exact same format
as the existing index (``quant_fund_agent.schemas.Paper``).

Usage:
    ./venv/bin/python scripts/populate_papers_hft.py [--max-papers N] [--dry-run]

Defaults aim to add ~500-600 new papers across the 30 queries below.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("populate_papers_hft")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PAPER_INDEX_PATH = Path(os.getenv("PAPER_INDEX_PATH", "data/papers/index.json"))

SEARCH_QUERIES = [
    # Stochastic Processes & Rough Path Theory
    "signature transforms limit order book alpha",
    "rough path theory high frequency price prediction",
    "marked Hawkes process order flow toxicity",
    "rough volatility fractional Brownian motion tick data",
    "self-exciting point processes order book imbalance",

    # Deep Learning & Advanced Neural Architectures
    "limit order book transformer self-attention microstructure",
    "DeepLOB convolutional neural networks order book alpha",
    "graph convolutional networks dynamic time warping cross-asset LOB",
    "reservoir computing echo state networks high frequency alpha",
    "reinforcement learning optimal market making L3 data",

    # Microstructure Dynamics & Liquidity State
    "volume-synchronized probability of toxicity VPIN adverse selection",
    "queue position cancellation-to-fill ratio alpha",
    "order flow imbalance tick-by-tick cross-impact",
    "latent liquidity iceberg order detection L3 data",
    "intraday lead-lag effects information leadership order book",

    # Quantitative Execution & Market Impact
    "Avellaneda-Stoikov inventory risk management high frequency",
    "transient market impact Hasbrouck microstructure alpha",
    "volume profile TWAP deviations mean reversion tick data",
    "Fourier transforms wavelet decomposition regime detection order book",
    "latency arbitrage adverse selection L2 quote stuffing",

    # Advanced Mathematics & Physics-Based Modeling
    "order book shape dynamics partial differential equations",
    "stochastic control limit order placement L3 data",
    "cross-impact models propagator limit order book",
    "fractional cointegration high frequency statistical arbitrage",
    "micro-price limit order book imbalance fair value estimation",

    # Ultra-High-Frequency Stochastics & Regime Modeling
    "hidden Markov models regime switching high frequency liquidity",
    "point process models trade duration ultra high frequency",
    "cancellation dynamics limit order book resilience",
    "market microstructure noise robust volatility estimation",
    "Hawkes processes cross-asset market impact tick data",
    "information asymmetry probability of informed trading PIN L2 data",
]

LLM_MODEL = os.getenv("POPULATE_PAPERS_LLM_MODEL", "gpt-4o-mini")
# Fetched per query before cross-query dedup / cap.  High enough that, even
# after removing the ~446 papers already in the DB and inter-query overlap,
# the 30 queries still yield 500-600 unique new papers.
ARXIV_MAX_RESULTS_PER_QUERY = 100
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
# Helpers
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


def _bare_arxiv_id(arxiv_id: str) -> str:
    """Drop the version suffix so 'v2'/'v3' of the same paper collide."""
    return re.sub(r"v\d+$", "", arxiv_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate paper DB with HFT/microstructure papers from ArXiv")
    parser.add_argument("--max-papers", type=int, default=600,
                        help="Maximum number of new papers to add (default: 600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing anything or calling the LLM")
    args = parser.parse_args()

    # -- Load existing DB ------------------------------------------------
    from quant_fund_agent.databases import PaperDatabase
    from quant_fund_agent.schemas import Paper, PaperStatus

    paper_db = PaperDatabase()
    paper_db.load_from_json(PAPER_INDEX_PATH)
    existing_ids = {p.id for p in paper_db.list_papers()}
    # Seed the dedup set with bare (version-stripped) IDs already in the DB so
    # no paper is ever added twice, even across script runs / arxiv versions.
    seen_arxiv: set[str] = {_bare_arxiv_id(a) for a in _load_existing_arxiv_ids(paper_db)}
    log.info("Existing papers in DB: %d  (existing arxiv ids: %d)", len(existing_ids), len(seen_arxiv))

    # -- Collect candidates from ArXiv -----------------------------------
    candidates: list[dict] = []

    for query in SEARCH_QUERIES:
        if len(candidates) >= args.max_papers:
            log.info("Reached candidate target (%d); stopping search.", args.max_papers)
            break
        log.info("Searching: %s", query)
        try:
            results = _fetch_arxiv(query, max_results=ARXIV_MAX_RESULTS_PER_QUERY)
        except Exception as e:
            log.warning("ArXiv query failed ('%s'): %s", query, e)
            continue
        added = 0
        for r in results:
            bare = _bare_arxiv_id(r["arxiv_id"])
            if bare not in seen_arxiv:
                seen_arxiv.add(bare)
                candidates.append(r)
                added += 1
        log.info("  -> +%d new (%d unique candidates total)", added, len(candidates))
        time.sleep(3.0)  # avoid ArXiv 429s between queries

    candidates = candidates[: args.max_papers]
    log.info("Total new candidates to process: %d", len(candidates))

    if args.dry_run:
        for c in candidates[:30]:
            print(f"  [{c['arxiv_id']}] {c['title'][:80]}")
        if len(candidates) > 30:
            print(f"  ... and {len(candidates) - 30} more")
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

        # Save incrementally so progress isn't lost on interrupt
        if i % 25 == 0:
            paper_db.save_to_json(PAPER_INDEX_PATH)
            log.info("  Checkpoint saved (%d papers written so far)", i)

    paper_db.save_to_json(PAPER_INDEX_PATH)
    log.info("Done. Added %d new papers. Total in DB: %d",
             len(new_papers), len(paper_db.list_papers()))


if __name__ == "__main__":
    main()
