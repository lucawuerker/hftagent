"""Deterministic backends for the Factor Researcher, shared by the
quant-research MCP server and its in-process fallback.

The Factor Researcher's LLM steps (brainstorm, codegen) stay in the agent graph;
everything *deterministic* it does — selecting/reading papers, deduping against
existing factors, materialising LLM-generated code (validate/write/import/smoke
test), running the single-factor IC backtest, and persisting/rejecting survivors
— lives here so the MCP server is a thin protocol wrapper and the in-process
fallback computes identical results.

The factor panel is cached at module level (keyed by field-set + the full
effective data config) so a long-lived server process loads the heavy intraday
panel once and reuses it across every candidate's IC backtest — while a switch of
provider / asset-class / universe / date-range correctly invalidates the cache.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("research.service")

PAPER_INDEX_PATH = Path(os.getenv("PAPER_INDEX_PATH", "data/papers/index.json"))
PAPER_PDF_DIR = Path(os.getenv("PAPER_PDF_DIR", "data/papers/pdfs"))


def _factor_db_path() -> Path:
    """Where research persists factors — resolved at call time.

    Read live from ``FACTOR_DB_PATH`` on every call (rather than captured at
    import) so the walk-forward backtest can redirect persistence to a
    run-scoped DB without the module having to be re-imported.
    """
    return Path(os.getenv("FACTOR_DB_PATH", "data/factors/factor_db.json"))


def _read_log_path() -> Path | None:
    """Scenario-scoped "papers already read" log, or ``None`` if unset.

    When ``PAPER_READ_LOG`` is set (the prerun and each backtest point it at
    their own file), :func:`load_papers` excludes ids already in the log and
    appends the ids it reads — so the prerun and a backtest never bias each
    other's paper selection and the global paper index is never mutated.
    Unset → today's behaviour (no exclusion, no write).
    """
    raw = os.getenv("PAPER_READ_LOG")
    return Path(raw) if raw else None

# Most papers in the index have no local PDF — only a link (arXiv abstract page +
# a direct ``pdf_url``).  We fetch and cache their *full text* from that link so
# the agent reads the whole paper, not just the stored abstract.
PAPER_FULLTEXT_CACHE_DIR = Path(
    os.getenv("PAPER_FULLTEXT_CACHE", "data/papers/fulltext_cache")
)
# Set PAPER_FETCH_FULLTEXT=0 to stay fully offline (abstract-only fallback).
FETCH_FULLTEXT = os.getenv("PAPER_FETCH_FULLTEXT", "1").strip().lower() not in (
    "0", "false", "no",
)
_HTTP_TIMEOUT = float(os.getenv("PAPER_FETCH_TIMEOUT", "30"))
_HTTP_HEADERS = {"User-Agent": "QuantFundAgent/1.0 (academic factor research)"}

# Panel cache keyed by (field-set, effective data config); see the original
# factor_research graph for why memory makes this cache worthwhile.  The config
# half of the key spans every ``DataSettings`` dimension that changes panel
# content (provider, asset class, frequency, universe/tickers, start/end, dtype,
# …), so a config switch never hands back a stale panel.
_PANEL_CACHE: dict[tuple[frozenset[str], tuple], dict[str, Any]] = {}


def _parse_cutoff(cutoff_date: str | None) -> date | None:
    if not cutoff_date:
        return None
    return date.fromisoformat(cutoff_date)


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

def _select_paper_ids(
    paper_db,
    n: int,
    cutoff: date | None,
    strategy: str,
    exclude: set[str] | None = None,
) -> list[str]:
    import random

    if cutoff is not None:
        papers = paper_db.list_papers_before(cutoff)
    else:
        papers = paper_db.list_papers()
    if exclude:
        papers = [p for p in papers if p.id not in exclude]
    if not papers:
        return []
    if strategy == "random":
        random.shuffle(papers)
    else:  # "unread_first"
        papers.sort(key=lambda p: (p.status.value != "unread", p.title))
    return [p.id for p in papers[:n]]


def _load_read_log(path: Path | None) -> set[str]:
    """Load the set of already-read paper ids from a read-log file."""
    if path is None or not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
        return set(payload.get("read", []))
    except Exception:  # a corrupt log must not abort a session
        return set()


def _append_read_log(path: Path | None, already: set[str], new_ids: list[str]) -> None:
    """Persist ``already ∪ new_ids`` back to the read-log file (best-effort)."""
    if path is None or not new_ids:
        return
    merged = sorted(already | set(new_ids))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"read": merged}, indent=2))
    except Exception as e:  # pragma: no cover — read-log is best-effort
        log.debug("could not write read log %s: %s", path, e)


def _pdf_url_for(paper) -> str | None:
    """Best-effort direct-PDF (or fallback) URL for a paper's coupled link.

    Prefers the stored ``metadata.pdf_url``; otherwise derives the PDF link
    from an arXiv abstract URL, or uses the raw ``url`` as a last resort.
    """
    meta = paper.metadata or {}
    pdf_url = meta.get("pdf_url")
    if pdf_url:
        return pdf_url
    url = paper.url or ""
    if not url:
        return None
    m = re.search(r"arxiv\.org/abs/([^\s/?#]+)", url)
    if m:
        return f"https://arxiv.org/pdf/{m.group(1)}"
    return url


def _strip_html(html: str) -> str:
    """Crude HTML → text (drop scripts/styles/tags, collapse whitespace)."""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_fulltext(paper, max_chars: int | None) -> str:
    """Download + extract the full text of ``paper`` from its coupled link.

    The *complete* extracted text is cached under
    :data:`PAPER_FULLTEXT_CACHE_DIR` keyed by paper id, so each paper is
    fetched at most once and ``max_chars`` is applied only as a read-time
    view — raising the cap later never needs a cache rebuild.  Returns
    ``""`` on any failure (the caller then falls back to the stored
    abstract), and never raises — a single unreachable link must not
    abort a research session.
    """
    cache = PAPER_FULLTEXT_CACHE_DIR / f"{paper.id}.txt"
    if cache.exists():
        cached = cache.read_text(encoding="utf-8", errors="ignore")
        if cached.strip():
            return cached[:max_chars] if max_chars else cached

    if not FETCH_FULLTEXT:
        return ""

    url = _pdf_url_for(paper)
    if not url:
        return ""

    try:
        import requests

        from quant_fund_agent.utils.pdf import extract_text_from_bytes

        t0 = time.time()
        resp = requests.get(url, timeout=_HTTP_TIMEOUT, headers=_HTTP_HEADERS)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        # Extract/cache the WHOLE paper (max_chars=None); the cap is a
        # read-time view applied below, not baked into the cache.
        if "pdf" in ctype or url.lower().endswith(".pdf") or "/pdf/" in url.lower():
            text = extract_text_from_bytes(resp.content, max_chars=None)
        else:
            text = _strip_html(resp.text)
        log.info("fetched full text for %s from %s (%d chars, %.1fs)",
                 paper.id, url, len(text), time.time() - t0)
    except Exception as e:  # noqa: BLE001 — degrade to abstract on any failure
        log.warning("full-text fetch failed for %s (%s): %s", paper.id, url, e)
        return ""

    if text.strip():
        try:
            PAPER_FULLTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        except Exception as e:  # pragma: no cover — caching is best-effort
            log.debug("could not cache full text for %s: %s", paper.id, e)
    return text[:max_chars] if max_chars else text


def load_papers(
    n: int = 2,
    cutoff_date: str | None = None,
    strategy: str = "unread_first",
    max_chars: int = 200_000,  # whole paper, not just the beginning
) -> list[dict[str, Any]]:
    """Select papers (lookahead-safe) and extract their text.

    Returns a list of snippet dicts (``paper_id``, ``title``,
    ``published_date``, ``text``, ``file_path``).
    """
    from quant_fund_agent.databases import PaperDatabase
    from quant_fund_agent.utils.pdf import extract_text

    paper_db = PaperDatabase()
    paper_db.load_from_json(PAPER_INDEX_PATH)
    # Self-heal: register PDFs dropped in without index entries.
    paper_db.auto_discover_pdfs(PAPER_PDF_DIR, index_path=PAPER_INDEX_PATH)

    # Scenario-scoped "already read" exclusion (prerun vs. backtest): exclude
    # papers this scenario has already read so successive sessions advance
    # through the library without re-reading, and record the new reads back.
    read_log = _read_log_path()
    already_read = _load_read_log(read_log)
    paper_ids = _select_paper_ids(paper_db, n=n, cutoff=_parse_cutoff(cutoff_date),
                                  strategy=strategy, exclude=already_read)
    _append_read_log(read_log, already_read, paper_ids)

    snippets: list[dict[str, Any]] = []
    for pid in paper_ids:
        p = paper_db.get_paper(pid)
        if p is None:
            continue

        text = ""
        source = ""

        # 1) Prefer a locally-downloaded PDF.
        if p.file_path:
            path = PAPER_PDF_DIR.parent / p.file_path
            if path.exists():
                text = extract_text(path, max_chars=max_chars)
                if text:
                    source = "local_pdf"

        # 2) Otherwise fetch the *full text* from the paper's coupled link
        #    (most index.json papers are arXiv entries with a pdf_url).
        if not text:
            fetched = _fetch_fulltext(p, max_chars)
            if fetched:
                text = fetched
                source = "link"

        # 3) Last resort: the stored abstract + description.
        if not text:
            parts = []
            desc = p.metadata.get("description", "")
            if desc:
                parts.append(desc)
            if p.abstract:
                parts.append(p.abstract)
            text = "\n\n".join(parts)[:max_chars]
            if text:
                source = "abstract"

        if not text:
            log.warning("no text available for paper %s", pid)

        snippets.append({
            "paper_id": p.id,
            "title": p.title,
            "published_date": p.published_date.isoformat() if p.published_date else None,
            "text": text,
            "file_path": p.file_path or "",
            "source": source,
        })
    return snippets


# ---------------------------------------------------------------------------
# Existing factor IDs (for brainstorm dedupe)
# ---------------------------------------------------------------------------

def existing_factor_ids(scope: str = "package") -> list[str]:
    """Factor ids a new brainstorm should avoid, per ``scope``.

    - ``"package"`` (default): every registered factor class (all code in the
      shared researcher package) ∪ the ids in the active factor DB.
    - ``"prerun"``: only the ids in the active factor DB (``FACTOR_DB_PATH``), so
      a prerun's brainstorm is not anchored by other preruns' factors.  Safe
      because :func:`materialise_factor` refuses to overwrite an already-registered
      id, so a slipped-through clash is dropped rather than corrupting the package.
    """
    ids: set[str] = set()
    if scope != "prerun":
        from quant_fund_agent.factors import discover_factors, get_all_factor_classes

        discover_factors()
        ids.update(get_all_factor_classes().keys())

    factor_db_path = _factor_db_path()
    if factor_db_path.exists():
        try:
            payload = json.loads(factor_db_path.read_text())
            ids.update(f["id"] for f in payload.get("factors", []))
        except Exception:
            pass
    return sorted(ids)


# ---------------------------------------------------------------------------
# Codegen materialisation
# ---------------------------------------------------------------------------

def materialise_factor(factor_id: str, code: str) -> dict[str, Any]:
    """Validate, write, import and smoke-test LLM-generated factor code.

    Returns ``{"ok": True, "code_path": str}`` on success or
    ``{"ok": False, "error": str}`` on any validation/materialisation failure.
    """
    from quant_fund_agent.agents.factor_research.codegen import (
        CodeValidationError,
        materialise,
    )

    try:
        path = materialise(factor_id, code)
    except CodeValidationError as e:
        return {"ok": False, "error": f"validation failed: {e}"}
    except Exception as e:  # noqa: BLE001 — surface any failure to the retry prompt
        return {"ok": False, "error": f"materialisation failed: {e}"}
    return {"ok": True, "code_path": str(path)}


# ---------------------------------------------------------------------------
# IC backtest
# ---------------------------------------------------------------------------

def _required_fields(factor_ids: list[str]) -> list[str]:
    from quant_fund_agent.factors import instantiate_factor

    needed: set[str] = {"close"}
    for fid in factor_ids:
        try:
            factor = instantiate_factor(fid)
        except Exception:
            continue
        needed.update(getattr(factor, "inputs", []) or [])
    if needed == {"close"}:
        needed.update({"open", "high", "low", "volume"})
    return sorted(needed)


def _panel_cache_key(data_dir: str, fields: list[str],
                     n_tickers: int | None) -> tuple[frozenset[str], tuple]:
    """Cache key spanning *every* config dimension that changes panel content.

    ``load_panel`` re-resolves the active data config (``quant.config.yaml`` +
    env) on every call, so the panel it returns depends on far more than
    ``(fields, n_tickers)`` — the provider, asset class, frequency,
    universe/tickers, start/end and dtype all change it.  We fold the *effective*
    :class:`~quant_fund_agent.config.DataSettings` (overridden exactly the way
    ``load_panel`` overrides it) into the key, so a provider/config switch can
    never reuse a stale panel while an identical repeated call still hits cache.
    """
    from quant_fund_agent.config import get_settings

    settings = get_settings().with_data_overrides(
        data_dir=data_dir, n_tickers=n_tickers)
    parts: list[tuple[str, Any]] = []
    for f in dataclasses.fields(settings.data):
        v = getattr(settings.data, f.name)
        parts.append((f.name, tuple(v) if isinstance(v, list) else v))
    return (frozenset(fields), tuple(parts))


def _load_panel_cached(data_dir: str, fields: list[str],
                       n_tickers: int | None) -> dict[str, Any]:
    key = _panel_cache_key(data_dir, fields, n_tickers)
    if key not in _PANEL_CACHE:
        from quant_fund_agent.data import load_panel
        log.info("Loading panel from %s (fields=%s, n_tickers=%s) …",
                 data_dir, fields, n_tickers if n_tickers is not None else "all")
        t0 = time.time()
        _PANEL_CACHE[key] = load_panel(data_dir, fields=fields, n_tickers=n_tickers)
        log.info("Panel loaded in %.1fs (%d tickers, %d fields)",
                 time.time() - t0, _PANEL_CACHE[key]["close"].shape[1],
                 len(_PANEL_CACHE[key]))
    return _PANEL_CACHE[key]


def _slice_panel_to_cutoff(panel: dict[str, Any], cutoff: date | None) -> dict[str, Any]:
    if cutoff is None:
        return panel
    cutoff_ts = datetime.combine(cutoff, datetime.min.time())
    return {k: df.loc[df.index < cutoff_ts] for k, df in panel.items()}


def backtest_factors(
    factor_ids: list[str],
    horizon: int = 6,
    cutoff_date: str | None = None,
    data_dir: str = "ticker_data",
    n_tickers: int | None = 15,
) -> dict[str, Any]:
    """Run the standard single-factor IC backtest for each ``factor_id``.

    Returns ``{factor_id: {"ok": True, "metrics": {...}}}`` or
    ``{factor_id: {"ok": False, "error": str}}``.  ``metrics`` is a
    ``BacktestMetrics`` model dump.
    """
    from quant_fund_agent.backtesting.engine import backtest_factor
    from quant_fund_agent.factors import discover_factors, instantiate_factor

    discover_factors()
    if not factor_ids:
        return {}

    fields = _required_fields(factor_ids)
    panel_full = _load_panel_cached(data_dir, fields, n_tickers=n_tickers)
    panel = _slice_panel_to_cutoff(panel_full, _parse_cutoff(cutoff_date))

    # Capability-gating: a factor needing fields the active provider can't supply
    # is reported as gated up front, with a clear reason, instead of reaching a
    # cryptic missing-column error inside the backtest.  No-op on LOBSTER.
    available = frozenset(panel.keys())

    results: dict[str, Any] = {}
    for fid in factor_ids:
        try:
            factor = instantiate_factor(fid)
            from quant_fund_agent.data.tiers import is_compatible, required_tier

            inputs = list(getattr(factor, "inputs", ["close"]) or ["close"])
            if not is_compatible(inputs, available):
                tier = required_tier(inputs)
                log.info("[%s] gated: requires %s tier (provider lacks fields)", fid, tier)
                results[fid] = {"ok": False, "error": f"gated: requires {tier} tier"}
                continue
            t0 = time.time()
            # Anchor the headline IC at the factor's *own* prediction horizon
            # (still records the shared grid for reference).  Falls back to the
            # session reference horizon for legacy factors that declare none.
            fh = int(getattr(factor, "prediction_horizon", horizon) or horizon)
            metrics = backtest_factor(factor, panel, horizon=horizon, factor_horizon=fh)
            log.info("[%s] backtest %.1fs IC@%d=%s ICIR@%d=%s", fid,
                     time.time() - t0, fh, metrics.information_coefficient,
                     fh, metrics.ic_ir)
            results[fid] = {"ok": True, "metrics": metrics.model_dump(mode="json")}
        except Exception as e:  # noqa: BLE001 — one bad factor must not abort
            log.warning("[%s] backtest failed: %s", fid, e)
            results[fid] = {"ok": False, "error": f"backtest failed: {e}"}
    return results


# ---------------------------------------------------------------------------
# Persist / reject
# ---------------------------------------------------------------------------

def _drop_researcher_factor(factor_id: str, code_path: str) -> None:
    """Remove a rejected factor from the registry AND the filesystem."""
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY

    _FACTOR_REGISTRY.pop(factor_id, None)
    if code_path:
        try:
            Path(code_path).unlink(missing_ok=True)
        except Exception as e:  # pragma: no cover — best-effort cleanup
            log.debug("could not delete %s: %s", code_path, e)


def persist_results(
    kept_records: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Persist survivor ``FactorRecord``s and purge rejects.

    ``kept_records`` are ``FactorRecord`` model dumps; ``rejected`` is a list of
    ``{"factor_id": str, "code_path": str}``.  Writes the updated factor DB once.
    """
    from quant_fund_agent.config import get_settings, provenance_stamp
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorRecord

    factor_db_path = _factor_db_path()
    factor_db = FactorDatabase()
    factor_db.load_from_json(factor_db_path)

    # Stamp which (config, prerun) scope + data config this factor was researched
    # under, so a factor stays auditable after it is merged into a book.
    stamp = provenance_stamp(os.getenv("QF_SCOPE", "main"), get_settings().data)

    kept_ids: list[str] = []
    for raw in kept_records:
        record = FactorRecord.model_validate(raw)
        record.metadata.setdefault("provenance", stamp)
        factor_db.add_factor(record)
        kept_ids.append(record.id)

    rejected_ids: list[str] = []
    for item in rejected:
        fid = item["factor_id"]
        _drop_researcher_factor(fid, item.get("code_path", ""))
        rejected_ids.append(fid)

    factor_db.save_to_json(factor_db_path)
    log.info("research session persisted: %d kept, %d rejected",
             len(kept_ids), len(rejected_ids))
    return {"kept_factor_ids": kept_ids, "rejected_factor_ids": rejected_ids}
