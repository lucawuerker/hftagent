"""Named factor-research "preruns" — compatibility shim over :mod:`workspace`.

A *prerun* is one batch of factor research run with a chosen LLM.  Preruns are now
modularised by **(config, prerun)** under
``data/workspaces/<config>/preruns/<prerun>/`` (see :class:`quant_fund_agent.
workspace.Scope`).  This module preserves the original prerun-name-only API for
existing callers (the simulator and comparison harness), resolving a bare prerun
name to its scope when no legacy flat ``data/factors/preruns/<name>/`` dir exists.

Isolation is intentionally *lightweight*: generated ``.py`` code stays in the
shared ``factors/researcher/`` package (global id de-dup), and a prerun is scoped
purely by its factor DB — which is all the Selector's catalog reads
(``FACTOR_DB_PATH``).  ``build_downstream_factor_db`` composes the seed-only main
library + a prerun's researcher factors into the DB the downstream agents see.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorSource
from quant_fund_agent.workspace import MAIN_FACTOR_DB, Scope, list_scopes

log = logging.getLogger("factors.preruns")

PRERUNS_ROOT = Path("data/factors/preruns")          # legacy flat location
PAPER_READ_LOG_ROOT = Path("data/papers/preruns")    # legacy flat read-logs
# The SEED records are sourced from the (now seed-only) main library when
# composing a downstream DB.
BASE_FACTOR_DB = MAIN_FACTOR_DB


# ── prerun-name → scope resolution ─────────────────────────────────────────

def _scope_for_prerun(name: str) -> Scope | None:
    """The first (config, prerun) scope whose prerun matches ``name``, if any."""
    for sc in list_scopes():
        if sc.prerun == name and sc.factor_db_path.exists():
            return sc
    return None


# ── path helpers (legacy flat first, then scope) ───────────────────────────

def prerun_dir(name: str) -> Path:
    return PRERUNS_ROOT / name


def db_path(name: str) -> Path:
    legacy = PRERUNS_ROOT / name / "factor_db.json"
    if legacy.exists():
        return legacy
    sc = _scope_for_prerun(name)
    return sc.factor_db_path if sc else legacy


def manifest_path(name: str) -> Path:
    legacy = PRERUNS_ROOT / name / "manifest.json"
    if legacy.exists():
        return legacy
    sc = _scope_for_prerun(name)
    return sc.manifest_path if sc else legacy


def read_log_path(name: str) -> Path:
    legacy = PAPER_READ_LOG_ROOT / f"{name}_read_log.json"
    if legacy.exists():
        return legacy
    sc = _scope_for_prerun(name)
    return sc.read_log_path if sc else legacy


def list_preruns() -> list[str]:
    """Names of all preruns with a factor DB on disk (legacy flat ∪ scopes)."""
    names: set[str] = set()
    if PRERUNS_ROOT.exists():
        names |= {p.name for p in PRERUNS_ROOT.iterdir()
                  if (p / "factor_db.json").exists()}
    names |= {sc.prerun for sc in list_scopes()}
    return sorted(names)


# ── manifest ───────────────────────────────────────────────────────────────

def read_manifest(name: str) -> dict:
    p = manifest_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # a corrupt manifest must not abort a prerun
        return {}


def write_manifest(name: str, **fields) -> dict:
    """Merge ``fields`` into the prerun's manifest, stamping created/updated_at."""
    manifest = read_manifest(name)
    now = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("name", name)
    manifest.setdefault("created_at", now)
    manifest["updated_at"] = now
    manifest.update(fields)
    p = manifest_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def researcher_factor_count(name: str) -> int:
    """How many RESEARCHER factors the prerun's DB currently holds."""
    db = FactorDatabase()
    db.load_from_json(db_path(name))
    return len(db.list_factors(source=FactorSource.RESEARCHER))


# ── composing the downstream factor DB ─────────────────────────────────────

def build_downstream_factor_db(
    target_path: Path,
    prerun_name: str,
    include_seeds: bool = True,
    base_db: Path = BASE_FACTOR_DB,
) -> int:
    """Compose the factor DB the downstream agents see for one prerun.

    Writes ``target_path`` with the prerun's RESEARCHER factors, plus the SEED
    factors from ``base_db`` when ``include_seeds``.  Pointing ``FACTOR_DB_PATH``
    at the result scopes the Selector to exactly this set.  Returns the factor
    count written.
    """
    pdb = db_path(prerun_name)
    if not pdb.exists():
        available = list_preruns()
        raise FileNotFoundError(
            f"Prerun {prerun_name!r} has no factor DB at {pdb}. "
            f"Available preruns: {available or '(none)'}. "
            f"Create it with `run_factor_research.py --name {prerun_name} …`."
        )

    out = FactorDatabase()

    if include_seeds:
        base = FactorDatabase()
        base.load_from_json(base_db)
        for f in base.list_factors(source=FactorSource.SEED):
            out.add_factor(f)

    prerun = FactorDatabase()
    prerun.load_from_json(pdb)
    for f in prerun.list_factors(source=FactorSource.RESEARCHER):
        out.add_factor(f)

    n = len(out.list_factors())
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_json(target_path)
    log.info("Composed downstream factor DB for prerun '%s' (seeds=%s): %d factors → %s",
             prerun_name, include_seeds, n, target_path)
    return n


# ── teardown ───────────────────────────────────────────────────────────────

def targeted_purge(name: str) -> list[str]:
    """Delete a prerun: its dir + read-log, and only its own factor code files.

    Removes from the shared ``factors/researcher/`` package only the ``.py`` files
    whose ids belong to this prerun's DB (and drops them from the in-memory
    registry), so other preruns' and the seeds' code are never touched.  Resolves
    scope-based preruns via :meth:`Scope.purge`; otherwise purges the legacy flat
    layout.  Returns the purged factor ids.
    """
    legacy_db = PRERUNS_ROOT / name / "factor_db.json"
    if not legacy_db.exists():
        sc = _scope_for_prerun(name)
        if sc is not None:
            purged = sc.purge()
            log.info("Purged scope prerun '%s' (%d factor code files removed)",
                     sc.label, len(purged))
            return purged

    from quant_fund_agent.agents.factor_research.codegen import RESEARCHER_DIR
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY

    purged: list[str] = []
    if legacy_db.exists():
        db = FactorDatabase()
        db.load_from_json(legacy_db)
        for f in db.list_factors(source=FactorSource.RESEARCHER):
            (RESEARCHER_DIR / f"{f.id}.py").unlink(missing_ok=True)
            _FACTOR_REGISTRY.pop(f.id, None)
            purged.append(f.id)

    # Drop the legacy flat prerun dir (DB + manifest) and its read-log.
    import shutil

    if prerun_dir(name).exists():
        shutil.rmtree(prerun_dir(name), ignore_errors=True)
    legacy_log = PAPER_READ_LOG_ROOT / f"{name}_read_log.json"
    legacy_log.unlink(missing_ok=True)

    log.info("Purged prerun '%s' (%d factor code files removed)", name, len(purged))
    return purged
