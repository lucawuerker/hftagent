"""Whole-paper + chunk embeddings over the paper library, with cosine retrieval.

Design (per ``docs/research-evolution/DESIGN.md`` §RAG):

* **Two granularities.**  Whole-paper vectors do selection/ranking (which papers
  should ground this generation's ideas); chunk vectors do fine-grained
  grounding and citation checks (P4's GraphRAG also consumes them).  The *whole
  paper* is what gets passed to the LLM — retrieval is for routing, not for
  squeezing context.
* **Date-gating.**  Retrieval accepts a ``cutoff_date`` and only returns papers
  published strictly before it — the same no-look-ahead rule the paper loader
  already enforces.
* **Citation verification.**  :func:`verify_citations` — a claimed
  ``source_paper_id`` must be among the retrieved ids, else it is a
  hallucination and the caller drops it.
* **Scale.**  ~1k papers is tiny: an in-memory matrix + cosine is all we need.
  Embeddings are cached on disk (`.npz` + meta) keyed by a corpus fingerprint +
  embedder name, so the expensive embedding pass runs once.

Embedding backends: ``openai`` (``text-embedding-3-small``; needs
``OPENAI_API_KEY``) or ``hash`` — a deterministic, dependency-free
bag-of-token-hashes vectoriser that keeps everything runnable offline (tests,
CI, air-gapped smoke runs).  Select with ``QF_EMBEDDER`` or the constructor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

log = logging.getLogger("knowledge.embed_store")

DEFAULT_CACHE_DIR = Path(os.getenv("QF_EMBED_CACHE", "data/knowledge/embeddings"))
OPENAI_EMBED_MODEL = os.getenv("QF_EMBED_MODEL", "text-embedding-3-small")

# chunking: ~4k chars ≈ 1k tokens for dense text; generous overlap keeps a
# mechanism's description from being cut mid-sentence.
CHUNK_CHARS = 4_000
CHUNK_OVERLAP = 400


# ── embedders ────────────────────────────────────────────────────────────────

def hashing_embedder(texts: Sequence[str], dim: int = 512) -> np.ndarray:
    """Deterministic bag-of-token-hashes embedding (offline fallback).

    Each lowercase token is hashed into one of ``dim`` buckets with a signed
    contribution; vectors are L2-normalised.  Crude, but it ranks lexically
    similar texts together, which is enough for tests and offline smoke runs —
    and it is 100% reproducible with no key and no network.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0 if (h >> 60) % 2 == 0 else -1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def openai_embedder(texts: Sequence[str], model: str = OPENAI_EMBED_MODEL,
                    batch: int = 128) -> np.ndarray:
    """OpenAI embeddings, batched; unit-normalised."""
    from openai import OpenAI  # lazy: only needed when actually embedding

    client = OpenAI()
    vecs: list[list[float]] = []
    for lo in range(0, len(texts), batch):
        chunk = [t[:32_000] or " " for t in texts[lo:lo + batch]]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs.extend(d.embedding for d in resp.data)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def resolve_embedder(name: str | None = None,
                     ) -> tuple[str, Callable[[Sequence[str]], np.ndarray]]:
    """(embedder_name, callable) from an explicit name or ``QF_EMBEDDER``.

    Defaults to ``openai`` when a key is present, else ``hash``.
    """
    name = (name or os.getenv("QF_EMBEDDER") or "").strip().lower()
    if not name:
        name = "openai" if os.getenv("OPENAI_API_KEY") else "hash"
    if name == "openai":
        return "openai:" + OPENAI_EMBED_MODEL, openai_embedder
    if name == "hash":
        return "hash:v1", hashing_embedder
    raise ValueError(f"unknown embedder {name!r} (use 'openai' or 'hash')")


# ── corpus ───────────────────────────────────────────────────────────────────

@dataclass
class CorpusDoc:
    """One paper as the store sees it."""

    paper_id: str
    title: str = ""
    published_date: str | None = None   # ISO date
    text: str = ""
    # Harvest-block label ("price" / "fundamental" / "general"); None for
    # legacy papers indexed before scope tagging existed.
    data_scope: str | None = None

    @property
    def published(self) -> date | None:
        if not self.published_date:
            return None
        try:
            return date.fromisoformat(str(self.published_date)[:10])
        except ValueError:
            return None


def load_corpus(max_chars: int = 60_000) -> list[CorpusDoc]:
    """The paper library as embedding-ready docs (index + cached full text).

    Reads the paper index and, per paper, the *cached* full text if a previous
    session fetched it (falling back to abstract + description).  It never
    fetches from the network and never touches the read-log — building the
    embedding index must not bias paper selection anywhere else.
    """
    from quant_fund_agent.databases import PaperDatabase
    from quant_fund_agent.mcp.research_service import (
        PAPER_FULLTEXT_CACHE_DIR,
        PAPER_INDEX_PATH,
    )

    paper_db = PaperDatabase()
    paper_db.load_from_json(PAPER_INDEX_PATH)

    docs: list[CorpusDoc] = []
    for p in paper_db.list_papers():
        text = ""
        cache = PAPER_FULLTEXT_CACHE_DIR / f"{p.id}.txt"
        if cache.exists():
            text = cache.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            parts = [p.metadata.get("description", "") if p.metadata else "",
                     p.abstract or ""]
            text = "\n\n".join(s for s in parts if s)
        docs.append(CorpusDoc(
            paper_id=p.id,
            title=p.title or p.id,
            published_date=p.published_date.isoformat() if p.published_date else None,
            text=(p.title or "") + "\n\n" + text[:max_chars],
            data_scope=(p.metadata or {}).get("data_scope"),
        ))
    return docs


def _chunk_spans(text: str, size: int = CHUNK_CHARS,
                 overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """Contiguous (start, end) character spans covering ``text`` with overlap."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    step = max(1, size - overlap)
    for lo in range(0, len(text), step):
        hi = min(len(text), lo + size)
        spans.append((lo, hi))
        if hi >= len(text):
            break
    return spans


def _corpus_fingerprint(docs: Sequence[CorpusDoc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.paper_id):
        h.update(f"{d.paper_id}:{len(d.text)}".encode())
    return h.hexdigest()[:16]


# ── the store ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedPaper:
    paper_id: str
    title: str
    published_date: str | None
    score: float
    text: str = ""


@dataclass
class RetrievedChunk:
    paper_id: str
    chunk_index: int
    score: float
    text: str


class EmbedStore:
    """In-memory embedding matrices over the paper corpus, disk-cached."""

    def __init__(self, docs: Sequence[CorpusDoc] | None = None, *,
                 embedder: str | None = None,
                 cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.docs: list[CorpusDoc] = list(docs) if docs is not None else load_corpus()
        self.by_id = {d.paper_id: d for d in self.docs}
        self.embedder_name, self.embed = resolve_embedder(embedder)
        self.cache_dir = Path(cache_dir)
        self.paper_vecs: np.ndarray | None = None
        # chunk i ↔ (paper_id, chunk_index, (lo, hi))
        self.chunk_meta: list[tuple[str, int, tuple[int, int]]] = []
        self.chunk_vecs: np.ndarray | None = None

    # ── build / cache ──

    def _cache_paths(self) -> tuple[Path, Path]:
        return self.cache_dir / "embeddings.npz", self.cache_dir / "meta.json"

    def build(self, force: bool = False) -> "EmbedStore":
        """Embed papers + chunks (or load the disk cache if it still matches)."""
        npz_path, meta_path = self._cache_paths()
        fp = _corpus_fingerprint(self.docs)

        if not force and npz_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("fingerprint") == fp and \
                        meta.get("embedder") == self.embedder_name:
                    data = np.load(npz_path)
                    self.paper_vecs = data["papers"]
                    self.chunk_vecs = data["chunks"]
                    self.chunk_meta = [
                        (m[0], int(m[1]), (int(m[2]), int(m[3])))
                        for m in meta["chunk_meta"]]
                    log.info("embed cache hit: %d papers, %d chunks",
                             len(self.docs), len(self.chunk_meta))
                    return self
            except Exception as e:  # noqa: BLE001 — a bad cache is just rebuilt
                log.warning("embed cache unreadable (%s) — rebuilding", e)

        log.info("embedding %d papers (%s) …", len(self.docs), self.embedder_name)
        self.paper_vecs = self.embed([d.text for d in self.docs]) \
            if self.docs else np.zeros((0, 1), dtype=np.float32)

        chunk_texts: list[str] = []
        self.chunk_meta = []
        for d in self.docs:
            for ci, (lo, hi) in enumerate(_chunk_spans(d.text)):
                self.chunk_meta.append((d.paper_id, ci, (lo, hi)))
                chunk_texts.append(d.text[lo:hi])
        self.chunk_vecs = self.embed(chunk_texts) \
            if chunk_texts else np.zeros((0, 1), dtype=np.float32)

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, papers=self.paper_vecs,
                                chunks=self.chunk_vecs)
            meta_path.write_text(json.dumps({
                "fingerprint": fp,
                "embedder": self.embedder_name,
                "paper_ids": [d.paper_id for d in self.docs],
                # aligned with paper_ids; survives cache round-trips (adding
                # papers changes the fingerprint and rebuilds anyway)
                "data_scopes": [d.data_scope for d in self.docs],
                "chunk_meta": [[pid, ci, lo, hi]
                               for pid, ci, (lo, hi) in self.chunk_meta],
                "built": datetime.now().isoformat(timespec="seconds"),
            }))
        except Exception as e:  # noqa: BLE001 — caching is best-effort
            log.warning("could not write embed cache: %s", e)
        return self

    def _ensure_built(self) -> None:
        if self.paper_vecs is None:
            self.build()

    # ── retrieval ──

    def _date_mask(self, cutoff: str | date | None) -> np.ndarray:
        if cutoff is None:
            return np.ones(len(self.docs), dtype=bool)
        cut = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
        mask = np.zeros(len(self.docs), dtype=bool)
        for i, d in enumerate(self.docs):
            pub = d.published
            # Undated papers are excluded under a cutoff: we cannot prove they
            # existed before it, and look-ahead safety beats recall here.
            mask[i] = pub is not None and pub < cut
        return mask

    def _scope_mask(self, allowed_scopes: set[str] | None) -> np.ndarray:
        """True where the doc's ``data_scope`` is allowed.

        ``None`` (no filter) keeps everything.  Docs with ``data_scope=None``
        — the legacy papers indexed before scope tagging — always pass, so a
        scope filter never silences the pre-existing library.
        """
        if allowed_scopes is None:
            return np.ones(len(self.docs), dtype=bool)
        mask = np.zeros(len(self.docs), dtype=bool)
        for i, d in enumerate(self.docs):
            mask[i] = d.data_scope is None or d.data_scope in allowed_scopes
        return mask

    def retrieve_papers(self, query: str, k: int = 5, *,
                        cutoff_date: str | date | None = None,
                        exclude_ids: set[str] | None = None,
                        allowed_scopes: set[str] | None = None,
                        with_text: bool = True) -> list[RetrievedPaper]:
        """Top-``k`` papers by cosine similarity, date- and scope-gated."""
        self._ensure_built()
        if not self.docs:
            return []
        qv = self.embed([query])[0]
        sims = self.paper_vecs @ qv
        mask = self._date_mask(cutoff_date) & self._scope_mask(allowed_scopes)
        if exclude_ids:
            for i, d in enumerate(self.docs):
                if d.paper_id in exclude_ids:
                    mask[i] = False
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        out: list[RetrievedPaper] = []
        for i in order:
            if not np.isfinite(sims[i]):
                break
            d = self.docs[int(i)]
            out.append(RetrievedPaper(
                paper_id=d.paper_id, title=d.title,
                published_date=d.published_date, score=float(sims[i]),
                text=d.text if with_text else ""))
        return out

    def retrieve_chunks(self, query: str, k: int = 8, *,
                        cutoff_date: str | date | None = None,
                        allowed_scopes: set[str] | None = None) -> list[RetrievedChunk]:
        """Top-``k`` chunks by cosine similarity (grounding / citation checks).

        Chunks inherit the parent paper's ``data_scope`` — the mask is applied
        at the paper level and chunks of a masked-out paper never surface.
        """
        self._ensure_built()
        if self.chunk_vecs is None or not len(self.chunk_meta):
            return []
        qv = self.embed([query])[0]
        sims = self.chunk_vecs @ qv
        mask = self._date_mask(cutoff_date) & self._scope_mask(allowed_scopes)
        allowed_papers = {d.paper_id for i, d in enumerate(self.docs) if mask[i]}
        order = np.argsort(-sims)
        out: list[RetrievedChunk] = []
        for i in order:
            pid, ci, (lo, hi) = self.chunk_meta[int(i)]
            if pid not in allowed_papers:
                continue
            out.append(RetrievedChunk(paper_id=pid, chunk_index=ci,
                                      score=float(sims[i]),
                                      text=self.by_id[pid].text[lo:hi]))
            if len(out) >= k:
                break
        return out


# ── citation verification ────────────────────────────────────────────────────

def verify_citations(claimed_ids: Sequence[str],
                     retrieved_ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split claimed source ids into (verified, hallucinated).

    A claimed ``source_paper_id`` that was not in the retrieved set cannot have
    grounded the idea — it is a hallucination and the caller drops/flags it.
    """
    retrieved = set(retrieved_ids)
    verified = [c for c in claimed_ids if c in retrieved]
    hallucinated = [c for c in claimed_ids if c not in retrieved]
    return verified, hallucinated
