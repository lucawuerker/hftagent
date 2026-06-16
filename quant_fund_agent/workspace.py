"""Single source of truth for the on-disk layout of one *scope*.

The fund is strictly modularised by **(config, prerun)**:

- The canonical **main** factor library — ``data/factors/factor_db.json`` — holds
  *only* the pre-defined formulaic SEED alphas.  Research never writes it.
- Everything researched lives in a :class:`Scope` = one *data config* (provider /
  asset-class / universe / timespan) × one *prerun* (a batch of factors mined by
  one research LLM)::

      data/workspaces/<config>/                      # a data config
        config.snapshot.json                         # the DataSettings it pins (shared)
        preruns/<prerun>/                            # one research-LLM batch
          manifest.json
          papers_read.json                           # paper read-log
          factors/factor_db.json                     # this scope's RESEARCHER factors
          factors/_composed.json                     # seeds + researcher (runtime view)
          factors/code/                              # archival snapshot of the .py
          strategies/{strategy_db.json, returns/, models/}
          portfolio/portfolio_db.json
          showcase.json

- A :class:`Book` is a *composed* active book under ``data/books/<name>/`` that
  the ``merge`` tool fills by pooling selected scopes' factors + strategies for
  the Portfolio Manager to allocate across.  Main is never mutated.

Rather than scattering these paths across ``pipeline``/``run_fund``/``preruns``/
``strategy_db``/``train`` (as before), every consumer reads them from here.  The
``export_env`` / ``activate`` seam mirrors ``simulation/simulator.py``: the
agents and the MCP modeling subprocess pick up ``FACTOR_DB_PATH`` /
``PAPER_READ_LOG`` / ``STRATEGY_RETURNS_DIR`` / ``MODEL_ARTIFACT_DIR`` *live* from
the environment, so activating a scope isolates everything it writes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

WORKSPACES_ROOT = Path("data/workspaces")
BOOKS_ROOT = Path("data/books")
# The seed-only canonical library.  A literal (not ``pipeline.FACTOR_DB_PATH``)
# avoids an import cycle; ``pipeline`` and ``preruns`` both point here.
MAIN_FACTOR_DB = Path("data/factors/factor_db.json")


def compose_factor_db(
    target_path: Path,
    researcher_db: Path,
    *,
    include_seeds: bool = True,
    base_db: Path | None = None,
) -> int:
    """Write ``target_path`` = SEED factors (from main) + RESEARCHER factors.

    Pointing ``FACTOR_DB_PATH`` at the result scopes the Selector to exactly this
    set.  Returns the factor count written.  Lazily imports the DB so this module
    stays dependency-light.  ``base_db`` is resolved at call time (defaults to the
    module-level :data:`MAIN_FACTOR_DB`) so callers/tests can repoint the seed DB.
    """
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    base_db = base_db if base_db is not None else MAIN_FACTOR_DB
    out = FactorDatabase()
    if include_seeds and Path(base_db).exists():
        base = FactorDatabase()
        base.load_from_json(base_db)
        for f in base.list_factors(source=FactorSource.SEED):
            out.add_factor(f)
    if Path(researcher_db).exists():
        rdb = FactorDatabase()
        rdb.load_from_json(researcher_db)
        for f in rdb.list_factors(source=FactorSource.RESEARCHER):
            out.add_factor(f)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_json(target_path)
    return len(out.list_factors())


@dataclass(frozen=True)
class Scope:
    """A (config, prerun) pair — the unit of factor/strategy isolation."""

    config: str
    prerun: str
    root: Path = WORKSPACES_ROOT

    # ── identity ────────────────────────────────────────────────────────
    @property
    def label(self) -> str:
        return f"{self.config}/{self.prerun}"

    # ── config level (shared by every prerun under a config) ────────────
    @property
    def config_dir(self) -> Path:
        return self.root / self.config

    @property
    def config_snapshot_path(self) -> Path:
        return self.config_dir / "config.snapshot.json"

    # ── prerun level ────────────────────────────────────────────────────
    @property
    def dir(self) -> Path:
        return self.config_dir / "preruns" / self.prerun

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def read_log_path(self) -> Path:
        return self.dir / "papers_read.json"

    # factors
    @property
    def factor_db_path(self) -> Path:
        """This scope's RESEARCHER factors (what research persists)."""
        return self.dir / "factors" / "factor_db.json"

    @property
    def composed_db_path(self) -> Path:
        """Runtime view (seeds + this scope's researcher factors) the agents read."""
        return self.dir / "factors" / "_composed.json"

    @property
    def factor_code_dir(self) -> Path:
        """Archival snapshot of this scope's generated ``.py`` (code stays shared)."""
        return self.dir / "factors" / "code"

    # strategies
    @property
    def strategy_db_path(self) -> Path:
        return self.dir / "strategies" / "strategy_db.json"

    @property
    def returns_dir(self) -> Path:
        return self.dir / "strategies" / "returns"

    @property
    def models_dir(self) -> Path:
        return self.dir / "strategies" / "models"

    # portfolio / reporting
    @property
    def portfolio_db_path(self) -> Path:
        return self.dir / "portfolio" / "portfolio_db.json"

    @property
    def showcase_path(self) -> Path:
        return self.dir / "showcase.json"

    # ── lifecycle ───────────────────────────────────────────────────────
    def ensure(self) -> "Scope":
        for p in (self.factor_db_path.parent, self.factor_code_dir,
                  self.returns_dir, self.models_dir,
                  self.portfolio_db_path.parent):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def export_env(self) -> dict[str, str]:
        """The env seam every downstream service reads live (research uses the
        researcher-only DB; the composed view is for the downstream agents)."""
        return {
            "FACTOR_DB_PATH": str(self.factor_db_path),
            "PAPER_READ_LOG": str(self.read_log_path),
            "STRATEGY_RETURNS_DIR": str(self.returns_dir),
            "MODEL_ARTIFACT_DIR": str(self.models_dir),
            "QF_SCOPE": self.label,
        }

    def activate(self, *, factor_db: Path | None = None) -> None:
        """Stamp this scope's paths into ``os.environ`` (mirrors simulator.py).

        Must run *before* anything triggers the MCP modeling subprocess, which
        snapshots the environment at spawn time.  ``factor_db`` overrides which DB
        ``FACTOR_DB_PATH`` points at (e.g. the composed view for downstream agents
        vs. the researcher-only DB during research).
        """
        self.ensure()
        env = self.export_env()
        if factor_db is not None:
            env["FACTOR_DB_PATH"] = str(factor_db)
        os.environ.update(env)

    def compose_runtime_db(self, *, include_seeds: bool = True) -> int:
        """Build ``composed_db_path`` = seeds + this scope's researcher factors."""
        return compose_factor_db(
            self.composed_db_path, self.factor_db_path, include_seeds=include_seeds)

    # ── config snapshot ─────────────────────────────────────────────────
    def write_config_snapshot(self, data) -> dict:
        """Persist the active ``DataSettings`` (+ fingerprint) for reproducibility.

        Written once per config dir; a later run under a *different* config that
        reuses the same name is surfaced by :meth:`config_mismatch`.
        """
        import dataclasses as _dc

        from quant_fund_agent.config import config_fingerprint

        snap = {
            "config_hash": config_fingerprint(data),
            "config_name": self.config,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": _dc.asdict(data),
        }
        self.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_snapshot_path.exists():
            self.config_snapshot_path.write_text(json.dumps(snap, indent=2, default=str))
        return snap

    def read_config_snapshot(self) -> dict | None:
        if not self.config_snapshot_path.exists():
            return None
        try:
            return json.loads(self.config_snapshot_path.read_text())
        except Exception:  # a corrupt snapshot must not abort a run
            return None

    def config_mismatch(self, data) -> str | None:
        """Return a human warning if the live config differs from the snapshot."""
        from quant_fund_agent.config import config_fingerprint

        snap = self.read_config_snapshot()
        if not snap:
            return None
        live = config_fingerprint(data)
        if snap.get("config_hash") == live:
            return None
        return (
            f"Scope config {self.config!r} was created under config_hash "
            f"{snap.get('config_hash')!r} but the active config is {live!r}. "
            "Factors/strategies may have been researched on different data."
        )

    # ── prerun manifest ─────────────────────────────────────────────────
    def read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except Exception:  # a corrupt manifest must not abort a run
            return {}

    def write_manifest(self, **fields) -> dict:
        """Merge ``fields`` into the prerun manifest, stamping created/updated_at."""
        manifest = self.read_manifest()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        manifest.setdefault("config", self.config)
        manifest.setdefault("prerun", self.prerun)
        manifest.setdefault("created_at", now)
        manifest["updated_at"] = now
        manifest.update(fields)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        return manifest

    # ── teardown ────────────────────────────────────────────────────────
    def purge(self) -> list[str]:
        """Delete this scope's factors (DB + shared code + read-log) and drop the
        classes from the registry.  Other scopes and the seeds are untouched
        (factor ids are globally unique, so a scope owns its code files).
        """
        import shutil

        from quant_fund_agent.agents.factor_research.codegen import RESEARCHER_DIR
        from quant_fund_agent.databases import FactorDatabase
        from quant_fund_agent.factors.registry import _FACTOR_REGISTRY
        from quant_fund_agent.schemas import FactorSource

        purged: list[str] = []
        if self.factor_db_path.exists():
            db = FactorDatabase()
            db.load_from_json(self.factor_db_path)
            for f in db.list_factors(source=FactorSource.RESEARCHER):
                (RESEARCHER_DIR / f"{f.id}.py").unlink(missing_ok=True)
                _FACTOR_REGISTRY.pop(f.id, None)
                purged.append(f.id)
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        return purged


@dataclass(frozen=True)
class Book:
    """A composed active book under ``data/books/<name>`` (merge output)."""

    name: str
    root: Path = BOOKS_ROOT

    @property
    def dir(self) -> Path:
        return self.root / self.name

    @property
    def factor_db_path(self) -> Path:
        return self.dir / "factors" / "factor_db.json"

    @property
    def strategy_db_path(self) -> Path:
        return self.dir / "strategies" / "strategy_db.json"

    @property
    def returns_dir(self) -> Path:
        return self.dir / "strategies" / "returns"

    @property
    def models_dir(self) -> Path:
        return self.dir / "strategies" / "models"

    @property
    def portfolio_db_path(self) -> Path:
        return self.dir / "portfolio" / "portfolio_db.json"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def showcase_path(self) -> Path:
        return self.dir / "showcase.json"

    def ensure(self) -> "Book":
        for p in (self.factor_db_path.parent, self.returns_dir, self.models_dir,
                  self.portfolio_db_path.parent):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def export_env(self) -> dict[str, str]:
        return {
            "FACTOR_DB_PATH": str(self.factor_db_path),
            "STRATEGY_RETURNS_DIR": str(self.returns_dir),
            "MODEL_ARTIFACT_DIR": str(self.models_dir),
            "QF_SCOPE": f"book:{self.name}",
        }

    def activate(self) -> None:
        self.ensure()
        os.environ.update(self.export_env())


def extract_researcher_db(source: Path, target: Path) -> int:
    """Write ``target`` with only the RESEARCHER factors found in ``source``.

    Keeps a scope's canonical researcher DB (researcher-only) in sync after an
    inline research session persisted into the composed run DB (which also holds
    seeds).  Returns the researcher-factor count written.
    """
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    src = FactorDatabase()
    src.load_from_json(source)
    out = FactorDatabase()
    for f in src.list_factors(source=FactorSource.RESEARCHER):
        out.add_factor(f)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_json(target)
    return len(out.list_factors())


def list_scopes(root: Path = WORKSPACES_ROOT) -> list[Scope]:
    """Every (config, prerun) scope that has a factor DB on disk."""
    if not root.exists():
        return []
    out: list[Scope] = []
    for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
        preruns = cfg / "preruns"
        if not preruns.exists():
            continue
        for pr in sorted(p for p in preruns.iterdir() if p.is_dir()):
            out.append(Scope(cfg.name, pr.name, root=root))
    return out


def list_books(root: Path = BOOKS_ROOT) -> list[Book]:
    if not root.exists():
        return []
    return [Book(p.name, root=root) for p in sorted(root.iterdir()) if p.is_dir()]
