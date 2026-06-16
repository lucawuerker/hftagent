"""Compose an *active book* by pooling selected scopes' factors + strategies.

The canonical *main* factor library (``data/factors/factor_db.json``) stays
strictly seed-only and is **never** mutated.  Instead, ``merge_into_book`` pools
the researched factors (and, by default, the fitted strategies) of one or more
(config, prerun) :class:`Scope`\\s into a separate :class:`Book` under
``data/books/<name>/`` that the Portfolio Manager can allocate across — a
fund-of-funds view assembled deliberately, not by accident.

Properties:

- **Idempotent** — re-merging the same selection is a no-op (collisions resolved
  per ``on_collision``).
- **Auditable** — merged factors keep ``source=RESEARCHER`` and gain a
  ``metadata["merged"]`` stamp; strategies keep their ``provenance``.
- **Self-contained** — strategy ``.joblib`` artifacts and returns CSVs are copied
  into the book and their paths rewritten, and the factors backing every merged
  strategy are auto-included so the book's factor DB is closed under reference.

A strategy's artifact was fit on its source config's panel, so pooling strategies
from scopes with *different* config hashes is flagged (``config_hash_warnings``),
not blocked — that pooling is the caller's deliberate choice.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from quant_fund_agent.databases import FactorDatabase, StrategyDatabase
from quant_fund_agent.schemas import FactorSource
from quant_fund_agent.workspace import MAIN_FACTOR_DB, Book, Scope

log = logging.getLogger("merge")


@dataclass
class MergeReport:
    promoted_factor_ids: list[str] = field(default_factory=list)
    skipped_factor_ids: list[str] = field(default_factory=list)
    promoted_strategy_ids: list[str] = field(default_factory=list)
    skipped_strategy_ids: list[str] = field(default_factory=list)
    code_files_copied: list[str] = field(default_factory=list)
    artifacts_copied: list[str] = field(default_factory=list)
    returns_copied: list[str] = field(default_factory=list)
    config_hash_warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"factors +{len(self.promoted_factor_ids)} "
                f"(skipped {len(self.skipped_factor_ids)}), "
                f"strategies +{len(self.promoted_strategy_ids)} "
                f"(skipped {len(self.skipped_strategy_ids)}), "
                f"artifacts {len(self.artifacts_copied)}, "
                f"warnings {len(self.config_hash_warnings)}")


def _seed_factor_db() -> FactorDatabase:
    """A fresh DB pre-loaded with main's SEED factors (the book's foundation)."""
    db = FactorDatabase()
    if MAIN_FACTOR_DB.exists():
        main = FactorDatabase()
        main.load_from_json(MAIN_FACTOR_DB)
        for f in main.list_factors(source=FactorSource.SEED):
            db.add_factor(f)
    return db


def _source_researcher_factors(sources: list[Scope]) -> dict:
    """``factor_id -> (record, owning Scope)`` for every source's researcher DB."""
    out: dict = {}
    for sc in sources:
        if not sc.factor_db_path.exists():
            continue
        sdb = FactorDatabase()
        sdb.load_from_json(sc.factor_db_path)
        for f in sdb.list_factors(source=FactorSource.RESEARCHER):
            out.setdefault(f.id, (f, sc))
    return out


def merge_into_book(
    sources: list[Scope],
    book: Book,
    *,
    factor_ids: list[str] | None = None,
    include_strategies: bool = True,
    strategy_ids: list[str] | None = None,
    on_collision: str = "skip",
    dry_run: bool = False,
) -> MergeReport:
    """Pool selected researched factors (+ strategies) from ``sources`` into ``book``.

    ``factor_ids=None`` selects every researcher factor across ``sources``.
    ``on_collision`` ∈ {``"skip"`` (default, idempotent), ``"overwrite"``}.
    Validates that every merged strategy's backing factors are resolvable before
    writing anything (raises :class:`ValueError` otherwise), so a failed merge
    leaves the book untouched.
    """
    if on_collision not in ("skip", "overwrite"):
        raise ValueError(f"on_collision must be 'skip' or 'overwrite', got {on_collision!r}")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = MergeReport()

    # ── existing book factor DB (seed it from main on first merge) ──
    book_db = FactorDatabase()
    if book.factor_db_path.exists():
        book_db.load_from_json(book.factor_db_path)
    else:
        book_db = _seed_factor_db()

    available = _source_researcher_factors(sources)
    selected = list(available.keys()) if factor_ids is None else list(factor_ids)

    def _merge_factor(fid: str) -> bool:
        """Add one researcher factor to the book DB; return True if newly added."""
        if fid in book_db._factors and on_collision == "skip":
            if fid not in report.skipped_factor_ids and fid not in report.promoted_factor_ids:
                report.skipped_factor_ids.append(fid)
            return False
        if fid not in available:
            return False  # caller validates resolvability separately
        rec, sc = available[fid]
        rec = rec.model_copy(deep=True)
        rec.metadata = dict(rec.metadata or {})
        rec.metadata["merged"] = {
            "from_scope": sc.label,
            "merged_at": now,
            "original_config_hash": (rec.metadata.get("provenance") or {}).get("config_hash"),
        }
        book_db.add_factor(rec)
        if fid not in report.promoted_factor_ids:
            report.promoted_factor_ids.append(fid)
        return True

    # ── resolve strategies first so we can auto-include their backing factors ──
    seed_ids = {f.id for f in _seed_factor_db().list_factors()}

    strat_plan: list[tuple] = []  # (record, owning Scope)
    if include_strategies:
        for sc in sources:
            if not sc.strategy_db_path.exists():
                continue
            src_sdb = StrategyDatabase(returns_dir=sc.returns_dir)
            src_sdb.load_from_json(sc.strategy_db_path)
            for rec in src_sdb.list_strategies():
                if strategy_ids is not None and rec.id not in strategy_ids:
                    continue
                strat_plan.append((rec, sc))

        # Validate every strategy's backing factors are resolvable (book seeds,
        # already selected, or available in a source) before mutating anything.
        for rec, sc in strat_plan:
            for fid in rec.factor_ids:
                resolvable = (fid in seed_ids or fid in book_db._factors
                              or fid in available or fid in selected)
                if not resolvable:
                    raise ValueError(
                        f"strategy {rec.id!r} (from {sc.label}) references factor "
                        f"{fid!r} which is not a seed nor available in any source "
                        f"scope — merge the factor's scope too, or exclude the strategy."
                    )
            for fid in rec.factor_ids:  # auto-include researched backers
                if fid not in seed_ids and fid not in book_db._factors:
                    if fid not in selected:
                        selected.append(fid)

    # ── materialise factor selection ──
    for fid in selected:
        _merge_factor(fid)

    # ── strategies: copy artifacts + returns, rewrite paths, register ──
    book_sdb = StrategyDatabase(returns_dir=book.returns_dir)
    if book.strategy_db_path.exists():
        book_sdb.load_from_json(book.strategy_db_path)
    config_hashes: set[str] = set()

    for rec, sc in strat_plan:
        if rec.id in book_sdb._strategies and on_collision == "skip":
            report.skipped_strategy_ids.append(rec.id)
            continue
        ch = (rec.metadata or {}).get("provenance", {}).get("config_hash")
        if ch:
            config_hashes.add(ch)
        rec = rec.model_copy(deep=True)
        rec.metadata = dict(rec.metadata or {})
        rec.metadata["pooled_from"] = sc.label

        # artifact
        art = (rec.model_params or {}).get("model_artifact_path") or ""
        if art and Path(art).exists():
            dst = book.models_dir / Path(art).name
            if not dry_run:
                book.ensure()
                shutil.copy2(art, dst)
            rec.model_params = dict(rec.model_params)
            rec.model_params["model_artifact_path"] = str(dst)
            report.artifacts_copied.append(str(dst))

        # returns CSV
        src_returns = sc.returns_dir / f"{rec.id}.csv"
        rec.returns_path = str(book.returns_dir / f"{rec.id}.csv")
        if not dry_run:
            book.ensure()
            book_sdb.add_strategy(rec)
            if src_returns.exists():
                shutil.copy2(src_returns, book.returns_dir / f"{rec.id}.csv")
                report.returns_copied.append(str(book.returns_dir / f"{rec.id}.csv"))
        report.promoted_strategy_ids.append(rec.id)

    if len(config_hashes) > 1:
        report.config_hash_warnings.append(
            "pooling strategies fit under differing config hashes "
            f"{sorted(config_hashes)} — their artifacts were trained on different "
            "panels; cross-config pooling is your deliberate choice."
        )

    # ── copy researcher code that isn't already in the shared package ──
    if not dry_run:
        from quant_fund_agent.agents.factor_research.codegen import RESEARCHER_DIR
        for fid in report.promoted_factor_ids:
            if fid not in available:
                continue
            _, sc = available[fid]
            if (RESEARCHER_DIR / f"{fid}.py").exists():
                continue
            snap = sc.factor_code_dir / f"{fid}.py"
            if snap.exists():
                shutil.copy2(snap, RESEARCHER_DIR / f"{fid}.py")
                report.code_files_copied.append(f"{fid}.py")

    # ── persist ──
    if not dry_run:
        book.ensure()
        book_db.annotate_tiers()
        book_db.save_to_json(book.factor_db_path)
        book_sdb.flush_returns()
        book_sdb.save_to_json(book.strategy_db_path)
        _write_book_manifest(book, sources, report, config_hashes)

    log.info("merge → book %s: %s", book.name, report.summary())
    return report


def _write_book_manifest(book: Book, sources: list[Scope], report: MergeReport,
                         config_hashes: set[str]) -> None:
    manifest_path = book.manifest_path
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = {}
    pooled = sorted(set(manifest.get("pooled_scopes", [])) | {s.label for s in sources})
    manifest.update({
        "name": book.name,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pooled_scopes": pooled,
        "config_hashes": sorted(set(manifest.get("config_hashes", [])) | config_hashes),
        "n_factors": report and len(report.promoted_factor_ids),
    })
    manifest.setdefault("created_at", manifest["updated_at"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def parse_scope_ref(ref: str) -> Scope:
    """``"<config>/<prerun>"`` → :class:`Scope`."""
    if "/" not in ref:
        raise ValueError(f"scope ref must be '<config>/<prerun>', got {ref!r}")
    config, prerun = ref.split("/", 1)
    return Scope(config, prerun)
