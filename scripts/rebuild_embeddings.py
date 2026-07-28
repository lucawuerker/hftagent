"""Rebuild the paper-corpus embedding cache (FINAL_RUN_PLAN.md runbook step 2).

Embeds every paper (whole-paper + chunk vectors) with the chosen backend and
writes the disk cache under ``data/knowledge/embeddings`` (``QF_EMBED_CACHE``).
The evolutionary researcher's RAG/GraphRAG retrieval then loads that cache.

Usage::

    # the real thing (needs OPENAI_API_KEY; ~$5 for ~2k papers, minutes):
    ./venv/bin/python scripts/rebuild_embeddings.py --embedder openai --force

    # offline check of the plumbing (no key, no network, no cost):
    ./venv/bin/python scripts/rebuild_embeddings.py --embedder hash --force

Without ``--force`` the build is a no-op when the cache already matches the
current corpus fingerprint + embedder — safe to re-run after adding papers
(only a changed corpus re-embeds).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--embedder", choices=["openai", "hash"], default=None,
                   help="Embedding backend (default: QF_EMBEDDER env, else 'hash'). "
                        "The thesis runs use 'openai' (text-embedding-3-small).")
    p.add_argument("--force", action="store_true",
                   help="Re-embed even if the cache matches the current corpus.")
    p.add_argument("--max-chars", type=int, default=60_000,
                   help="Per-paper text cap fed to the embedder (default 60k).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")

    if args.embedder == "openai" and not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("--embedder openai needs OPENAI_API_KEY in .env/env")

    from quant_fund_agent.knowledge.embed_store import EmbedStore, load_corpus

    docs = load_corpus(max_chars=args.max_chars)
    if not docs:
        raise SystemExit("paper corpus is empty — run scripts/populate_papers.py first")

    store = EmbedStore(docs, embedder=args.embedder)
    n_chars = sum(len(d.text) for d in docs)
    print(f"corpus: {len(docs)} papers, {n_chars/1e6:.1f}M chars "
          f"(~{n_chars/4/1e6:.1f}M tokens) — embedder '{store.embedder_name}'")

    t0 = time.perf_counter()
    store.build(force=args.force)
    elapsed = time.perf_counter() - t0

    npz_path, meta_path = store._cache_paths()
    print(f"done in {elapsed:.1f}s: {store.paper_vecs.shape[0]} paper vectors, "
          f"{store.chunk_vecs.shape[0]} chunk vectors")
    print(f"cache: {npz_path} + {meta_path}")


if __name__ == "__main__":
    main()
