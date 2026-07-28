"""Populate data/papers/index.json with papers from ArXiv, in scoped blocks.

Searches ArXiv using named query *blocks* — each with its own query list,
category filter and per-query result count — generates a short description for
each paper via gpt-4o-mini, and upserts the results into the paper database.
Every harvested paper is stamped with ``metadata["data_scope"] = <block name>``
so retrieval can be scoped to the run's data feed (e.g. exclude the
``fundamental`` block when fundamentals are off).  Already-indexed papers
(matched by ArXiv ID stored in metadata["arxiv_id"]) are skipped so the script
is safe to re-run.

Blocks:
    price        — the original OHLCV/price-action quant queries (q-fin heavy).
    fundamental  — cross-sectional fundamental/quality-factor literature.
    general      — non-finance maths/ML/signal-processing that could inspire
                   novel factors (no q-fin category restriction).

Usage:
    ./venv/bin/python scripts/populate_papers.py \
        [--blocks fundamental,general,price] [--max-papers N] [--dry-run]
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

'''SEARCH_QUERIES = [
    # Stochastic Processes & Rough Path Theory
    "signature transforms limit order book alpha",
    "rough path theory high frequency price prediction",
    "marked Hawkes process order flow toxicity",
    "rough volatility fractional Brownian motion tick data",
    "self-exciting point processes order book imbalance",

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
]'''
# ── query blocks ───────────────────────────────────────────────────────────
# Each block: its own queries, its own arXiv category filter, and its own
# per-query result count.  The block name is stamped onto every harvested
# paper as metadata["data_scope"] so retrieval can be scoped per run.

PRICE_QUERIES = [
    # Advanced Time-Series & Volatility Stochastics
    "Rough volatility fractional Brownian motion OHLCV realized volatility estimation",
    "GARCH-MIDAS mixed data sampling macroeconomic variables daily OHLCV",
    "Fractional cointegration OHLCV data pairs trading statistical arbitrage",
    "Non-linear state-space models stochastic volatility OHLCV time series",
    "Multivariate Hawkes processes aggregated OHLCV volume-driven price jumps",

    # Geometric, Topological, & Wavelet Signal Processing
    "Topological Data Analysis persistence diagrams OHLCV regime switching",
    "Continuous Wavelet Transform scalogram OHLCV multi-resolution alpha",
    "Empirical Mode Decomposition Hilbert-Huang transform OHLCV trend extraction",
    "Visibility graph network analysis OHLCV time series stock market",
    "Fourier transform spectral density OHLCV cyclical intraday seasonality",

    # Deep Learning & Sequential Architectures
    "Temporal Fusion Transformer multi-horizon OHLCV forecasting attention",
    "Neural Ordinary Differential Equations Neural ODEs OHLCV continuous time series",
    "Informer Transformer long sequence time series forecasting OHLCV",
    "WaveNet dilated causal convolutions autoregressive OHLCV alpha generation",
    "Reservoir computing echo state networks multi-asset OHLCV pattern recognition",

    # Volume-Price Structural Dynamics & Microstructure Proxies
    "Volume-weighted average price VWAP deviations mean reversion OHLCV",
    "Garman-Klass vs Yang-Zhang volatility estimators OHLCV efficiency comparison",
    "Volume clock time-deformation OHLCV bar processing variance stabilization",
    "Price-Volume power law scaling relationship financial time series",
    "Information asymmetry proxies high-low spread estimator OHLCV data",

    # Statistical Machine Learning & Regime Detection
    "Hidden Markov Models Gaussian mixture regime switching OHLCV volatility",
    "Optimal transport Wasserstein distance OHLCV distribution shift detection",
    "Dynamic Time Warping DTW clustering cross-asset OHLCV lead-lag",
    "Change-point detection algorithms structural breaks OHLCV trend reversals",
    "Gradient boosted decision trees non-linear feature engineering OHLCV alpha",

    # Mathematical Control & Systematic Execution Strategy
    "Stochastic control Almgren-Chriss optimal liquidation OHLCV volume profile",
    "Reinforcement learning Q-learning trend following OHLCV action space",
    "Kelly criterion dynamic position sizing OHLCV regime filtering",
    "Markov decision process automated portfolio rebalancing daily OHLCV",
    "Bayesian structural time series BSTS counterfactual analysis OHLCV",
]

FUNDAMENTAL_QUERIES = [
    # Accruals / earnings quality
    "accruals anomaly earnings quality cross-sectional stock returns",
    "cash flow versus accruals earnings persistence stock return predictability",
    # Profitability / quality factors
    "gross profitability factor quality minus junk cross-sectional returns",
    "operating profitability return on equity factor investing",
    "Piotroski F-score fundamental strength value stocks returns",
    # Earnings surprise / drift / revisions
    "post-earnings announcement drift PEAD earnings surprise anomaly",
    "analyst forecast revisions earnings estimates stock return predictability",
    "standardized unexpected earnings SUE momentum cross-section",
    # Investment / asset growth
    "asset growth anomaly corporate investment cross-sectional returns",
    "investment factor capital expenditure abnormal returns CAPM",
    "net share issuance equity financing anomaly stock returns",
    # Intangibles / R&D
    "intangible capital R&D capitalization book-to-market mispricing",
    "research and development intensity innovation stock returns anomaly",
    # Distress / balance sheet
    "financial distress risk Ohlson O-score bankruptcy prediction returns",
    "leverage balance sheet strength default risk equity returns",
    # Low volatility with fundamentals
    "low volatility anomaly fundamentals profitability betting against beta",
    # Value / timing / fundamental momentum
    "value spread timing book-to-market factor premium predictability",
    "fundamental momentum earnings trends factor performance",
]

GENERAL_QUERIES = [
    # Point processes / state space
    "Hawkes process self-exciting point process inference",
    "state-space models Kalman filtering latent dynamics time series",
    # Signal processing / spectral
    "wavelet scattering transform invariant signal representation",
    "spectral graph methods graph signal processing learning",
    "compressed sensing sparse recovery signal reconstruction",
    # Random matrices / transport / topology
    "random matrix theory covariance denoising eigenvalue spectrum",
    "optimal transport Wasserstein distance distribution comparison",
    "topological data analysis persistent homology time series",
    # Extremes / breaks / uncertainty
    "extreme value theory tail estimation heavy tails",
    "change-point detection structural break online algorithms",
    "conformal prediction distribution-free uncertainty quantification",
    # Representation learning / dynamics
    "self-supervised representation learning time series contrastive",
    "Koopman operator dynamic mode decomposition nonlinear dynamics",
    "rough path theory signature features sequential data",
    "causal discovery time series Granger structure learning",
    "dynamical systems reservoir computing chaotic time series prediction",
]

LLM_MODEL = os.getenv("POPULATE_PAPERS_LLM_MODEL", "gpt-4o-mini")

# Categories where rough-path, Hawkes, and signature-transform papers are
# typically filed (the original OHLCV block's filter).
PRICE_CATEGORIES = [
    "q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.MF", "q-fin.CP",
    "econ.GN", "cs.LG", "stat.ML", "math.PR",
]
# Cross-sectional fundamental/quality literature lives in the q-fin + econ set.
FUNDAMENTAL_CATEGORIES = [
    "q-fin.PM", "q-fin.ST", "q-fin.GN", "econ.GN",
]
# Deliberately NO q-fin restriction: the point of this block is non-finance
# mathematics/ML/signal-processing that could inspire novel factors.
GENERAL_CATEGORIES = [
    "math.PR", "math.ST", "stat.ML", "cs.LG", "eess.SP", "math.DS",
]

# Per-block per-query result counts tuned so the two NEW blocks yield roughly
# +1000 papers combined after arXiv-id dedup:
#   fundamental: 18 queries × 35 ≈ 630 raw
#   general:     16 queries × 30 ≈ 480 raw   → ~1100 raw, ~1000 after dedup.
QUERY_BLOCKS: dict[str, dict] = {
    "fundamental": {
        "queries": FUNDAMENTAL_QUERIES,
        "categories": FUNDAMENTAL_CATEGORIES,
        "max_results_per_query": 35,
    },
    "general": {
        "queries": GENERAL_QUERIES,
        "categories": GENERAL_CATEGORIES,
        "max_results_per_query": 30,
    },
    "price": {
        "queries": PRICE_QUERIES,
        "categories": PRICE_CATEGORIES,
        "max_results_per_query": 30,
    },
}


# ---------------------------------------------------------------------------
# ArXiv search
# ---------------------------------------------------------------------------

def _fetch_arxiv(query: str, max_results: int,
                 categories: list[str]) -> list[dict]:
    import arxiv

    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
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


def collect_candidates(blocks: dict[str, dict],
                       seen_arxiv: set[str],
                       max_candidates: int,
                       fetch=_fetch_arxiv,
                       sleep=time.sleep) -> list[dict]:
    """Run every query of every block; return deduped candidate dicts.

    Each candidate is stamped with ``data_scope`` = the name of the block whose
    query found it first (dedup is by arXiv id across blocks — first block
    wins).  ``seen_arxiv`` is mutated in place.  ``fetch``/``sleep`` are
    injectable for tests.
    """
    candidates: list[dict] = []
    for block_name, block in blocks.items():
        for query in block["queries"]:
            if len(candidates) >= max_candidates:
                return candidates[:max_candidates]
            log.info("[%s] Searching: %s", block_name, query)
            try:
                results = fetch(query,
                                max_results=block["max_results_per_query"],
                                categories=block["categories"])
            except Exception as e:
                log.warning("ArXiv query failed ('%s'): %s", query, e)
                continue
            for r in results:
                if r["arxiv_id"] not in seen_arxiv:
                    seen_arxiv.add(r["arxiv_id"])
                    candidates.append({**r, "data_scope": block_name})
            log.info("  → %d unique new candidates so far", len(candidates))
            sleep(3.0)  # avoid ArXiv 429s between queries
    return candidates[:max_candidates]


def build_paper(candidate: dict, description: str, existing_ids: set[str]):
    """One candidate dict → a ``Paper`` with data_scope stamped in metadata.

    Assigns a slug id, suffixing on collision; ``existing_ids`` is mutated in
    place with the assigned id.
    """
    from quant_fund_agent.schemas import Paper, PaperStatus

    base_id = _slugify(candidate["title"])
    paper_id = base_id
    suffix = 1
    while paper_id in existing_ids:
        suffix += 1
        paper_id = f"{base_id}_{suffix}"
    existing_ids.add(paper_id)

    return Paper(
        id=paper_id,
        title=candidate["title"],
        authors=candidate["authors"],
        abstract=candidate["abstract"],
        published_date=candidate["published_date"],
        status=PaperStatus.UNREAD,
        file_path=None,
        url=candidate["url"],
        metadata={
            "arxiv_id": candidate["arxiv_id"],
            "pdf_url": candidate["pdf_url"],
            "description": description,
            "data_scope": candidate.get("data_scope"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate paper database from ArXiv")
    parser.add_argument("--blocks", type=str,
                        default=",".join(QUERY_BLOCKS),
                        help="Comma-separated query blocks to run "
                             f"(default: all = {','.join(QUERY_BLOCKS)})")
    parser.add_argument("--max-papers", type=int, default=1200,
                        help="Maximum number of new papers to add (default: 1200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be added without writing anything")
    args = parser.parse_args()

    block_names = [b.strip() for b in args.blocks.split(",") if b.strip()]
    unknown = [b for b in block_names if b not in QUERY_BLOCKS]
    if unknown:
        parser.error(f"unknown block(s) {unknown}; choose from {list(QUERY_BLOCKS)}")
    blocks = {name: QUERY_BLOCKS[name] for name in block_names}

    # -- Load existing DB ------------------------------------------------
    from quant_fund_agent.databases import PaperDatabase

    paper_db = PaperDatabase()
    paper_db.load_from_json(PAPER_INDEX_PATH)
    existing_arxiv_ids = _load_existing_arxiv_ids(paper_db)
    existing_ids = {p.id for p in paper_db.list_papers()}
    log.info("Existing papers in DB: %d", len(existing_ids))

    # -- Collect candidates from ArXiv -----------------------------------
    seen_arxiv: set[str] = set(existing_arxiv_ids)
    candidates = collect_candidates(blocks, seen_arxiv, args.max_papers)
    log.info("Total new candidates to process: %d", len(candidates))

    if args.dry_run:
        for c in candidates[:20]:
            print(f"  [{c['data_scope']}] [{c['arxiv_id']}] {c['title'][:80]}")
        if len(candidates) > 20:
            print(f"  ... and {len(candidates) - 20} more")
        return

    # -- Set up LLM ------------------------------------------------------
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

    # -- Generate descriptions and build Paper objects -------------------
    new_papers = []
    for i, c in enumerate(candidates, 1):
        log.info("[%d/%d] (%s) %s", i, len(candidates),
                 c.get("data_scope"), c["title"][:70])

        description = _generate_description(c["title"], c["abstract"], llm)
        paper = build_paper(c, description, existing_ids)
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
