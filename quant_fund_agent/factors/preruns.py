"""Named factor-research "preruns".

A *prerun* is one batch of factor research run with a chosen LLM (e.g. ~100
factors mined by ``gpt-4o-mini`` vs ~100 by another model), kept under its own
directory so the **same** downstream agents can be run on each and compared:

```
data/factors/preruns/<name>/factor_db.json   # this prerun's researcher factors
data/factors/preruns/<name>/manifest.json    # model/provider/counts/timestamps
data/papers/preruns/<name>_read_log.json     # prerun-scoped paper read-log
```

Isolation is intentionally *lightweight*: generated ``.py`` code stays in the
shared ``factors/researcher/`` package (global id de-dup), and a prerun is scoped
purely by its factor DB — which is all the Selector's catalog reads
(``FACTOR_DB_PATH``).  This module owns the prerun layout, the manifest, composing
the downstream factor DB, and tearing a prerun down again.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorSource

log = logging.getLogger("factors.preruns")

PRERUNS_ROOT = Path("data/factors/preruns")
PAPER_READ_LOG_ROOT = Path("data/papers/preruns")
# The seed/global library SEED records are sourced from here when composing a
# downstream DB.  A literal (not pipeline.FACTOR_DB_PATH) avoids an import cycle.
BASE_FACTOR_DB = Path("data/factors/factor_db.json")


# ── path helpers ───────────────────────────────────────────────────────────

def prerun_dir(name: str) -> Path:
    return PRERUNS_ROOT / name


def db_path(name: str) -> Path:
    return prerun_dir(name) / "factor_db.json"


def manifest_path(name: str) -> Path:
    return prerun_dir(name) / "manifest.json"


def read_log_path(name: str) -> Path:
    return PAPER_READ_LOG_ROOT / f"{name}_read_log.json"


def list_preruns() -> list[str]:
    """Names of all preruns that have a factor DB on disk."""
    if not PRERUNS_ROOT.exists():
        return []
    return sorted(p.name for p in PRERUNS_ROOT.iterdir()
                  if (p / "factor_db.json").exists())


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
    registry), so other preruns' and the seeds' code are never touched.  Returns
    the purged factor ids.
    """
    from quant_fund_agent.agents.factor_research.codegen import RESEARCHER_DIR
    from quant_fund_agent.factors.registry import _FACTOR_REGISTRY

    purged: list[str] = []
    p = db_path(name)
    if p.exists():
        db = FactorDatabase()
        db.load_from_json(p)
        for f in db.list_factors(source=FactorSource.RESEARCHER):
            (RESEARCHER_DIR / f"{f.id}.py").unlink(missing_ok=True)
            _FACTOR_REGISTRY.pop(f.id, None)
            purged.append(f.id)

    # Drop the prerun dir (DB + manifest) and its read-log.
    import shutil

    if prerun_dir(name).exists():
        shutil.rmtree(prerun_dir(name), ignore_errors=True)
    read_log_path(name).unlink(missing_ok=True)

    log.info("Purged prerun '%s' (%d factor code files removed)", name, len(purged))
    return purged
