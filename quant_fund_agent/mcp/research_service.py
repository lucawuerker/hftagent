"""Deterministic backends for the Factor Researcher, shared by the
quant-research MCP server and its in-process fallback.

The Factor Researcher's LLM steps (brainstorm, codegen) stay in the agent graph;
everything *deterministic* it does — selecting/reading papers, deduping against
existing factors, materialising LLM-generated code (validate/write/import/smoke
test), running the single-factor IC backtest, and persisting/rejecting survivors
— lives here so the MCP server is a thin protocol wrapper and the in-process
fallback computes identical results.

The factor panel is cached at module level (keyed by field-set + universe size)
so a long-lived server process loads the heavy intraday panel once and reuses it
across every candidate's IC backtest.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("research.service")

PAPER_INDEX_PATH = Path(os.getenv("PAPER_INDEX_PATH", "data/papers/index.json"))
PAPER_PDF_DIR = Path(os.getenv("PAPER_PDF_DIR", "data/papers/pdfs"))
FACTOR_DB_PATH = Path(os.getenv("FACTOR_DB_PATH", "data/factors/factor_db.json"))

# Panel cache keyed by (field-set, universe size); see the original
# factor_research graph for why memory makes this cache worthwhile.
_PANEL_CACHE: dict[tuple[frozenset[str], int | None], dict[str, Any]] = {}


def _parse_cutoff(cutoff_date: str | None) -> date | None:
    if not cutoff_date:
        return None
    return date.fromisoformat(cutoff_date)


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

def _select_paper_ids(paper_db, n: int, cutoff: date | None, strategy: str) -> list[str]:
    import random

    if cutoff is not None:
        papers = paper_db.list_papers_before(cutoff)
    else:
        papers = paper_db.list_papers()
    if not papers:
        return []
    if strategy == "random":
        random.shuffle(papers)
    else:  # "unread_first"
        papers.sort(key=lambda p: (p.status.value != "unread", p.title))
    return [p.id for p in papers[:n]]


def load_papers(
    n: int = 2,
    cutoff_date: str | None = None,
    strategy: str = "unread_first",
    max_chars: int = 30_000,
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

    paper_ids = _select_paper_ids(paper_db, n=n, cutoff=_parse_cutoff(cutoff_date),
                                  strategy=strategy)

    snippets: list[dict[str, Any]] = []
    for pid in paper_ids:
        p = paper_db.get_paper(pid)
        if p is None:
            continue
        path = PAPER_PDF_DIR.parent / (p.file_path or "")
        text = extract_text(path, max_chars=max_chars)
        if not text:
            log.warning("no text extracted from %s", p.file_path)
        snippets.append({
            "paper_id": p.id,
            "title": p.title,
            "published_date": p.published_date.isoformat() if p.published_date else None,
            "text": text,
            "file_path": p.file_path or "",
        })
    return snippets


# ---------------------------------------------------------------------------
# Existing factor IDs (for brainstorm dedupe)
# ---------------------------------------------------------------------------

def existing_factor_ids() -> list[str]:
    """Union of (1) registered factor classes and (2) factor_db.json IDs."""
    import json

    from quant_fund_agent.factors import discover_factors, get_all_factor_classes

    discover_factors()
    ids: set[str] = set(get_all_factor_classes().keys())
    if FACTOR_DB_PATH.exists():
        try:
            payload = json.loads(FACTOR_DB_PATH.read_text())
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


def _load_panel_cached(data_dir: str, fields: list[str],
                       n_tickers: int | None) -> dict[str, Any]:
    key = (frozenset(fields), n_tickers)
    if key not in _PANEL_CACHE:
        from quant_fund_agent.backtesting.data_loader import load_panel
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

    results: dict[str, Any] = {}
    for fid in factor_ids:
        try:
            factor = instantiate_factor(fid)
            t0 = time.time()
            metrics = backtest_factor(factor, panel, horizon=horizon)
            log.info("[%s] backtest %.1fs IC@%d=%s ICIR@%d=%s", fid,
                     time.time() - t0, horizon, metrics.information_coefficient,
                     horizon, metrics.ic_ir)
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
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorRecord

    factor_db = FactorDatabase()
    factor_db.load_from_json(FACTOR_DB_PATH)

    kept_ids: list[str] = []
    for raw in kept_records:
        record = FactorRecord.model_validate(raw)
        factor_db.add_factor(record)
        kept_ids.append(record.id)

    rejected_ids: list[str] = []
    for item in rejected:
        fid = item["factor_id"]
        _drop_researcher_factor(fid, item.get("code_path", ""))
        rejected_ids.append(fid)

    factor_db.save_to_json(FACTOR_DB_PATH)
    log.info("research session persisted: %d kept, %d rejected",
             len(kept_ids), len(rejected_ids))
    return {"kept_factor_ids": kept_ids, "rejected_factor_ids": rejected_ids}
