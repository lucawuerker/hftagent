"""Prerun factor injection — a cheap, deterministic stand-in for LLM research.

A walk-forward backtest normally holds a *research meeting* on a cadence, in which
the Factor Researcher LLM invents new factors (see ``pipeline.run_research_session``).
That is expensive (one LLM session per meeting) and non-deterministic.

This module offers an alternative *factor source*: instead of researching new
factors each meeting, draw them — a few at a time — from one or more **preruns**
(batches of factors already mined offline, living under
``data/workspaces/<config>/preruns/<prerun>/``).  The effect is a fund whose factor
universe grows over the backtest exactly as it would under periodic research, but
sourced from a fixed, reproducible pool with **no LLM cost**.

The factor *code* for every prerun factor already lives in the shared
``factors/researcher/`` package (preruns keep their generated ``.py`` there for
global id de-dup — see :mod:`quant_fund_agent.factors.preruns`), so
``discover_factors`` registers them and their signals compute normally.  All this
module moves around is the factor *records* (catalog metadata) the Selector reads
from ``FACTOR_DB_PATH``.

Used by :func:`quant_fund_agent.simulation.simulator.run_backtest` when
``BacktestConfig.factor_source == "prerun_inject"``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from quant_fund_agent.databases import FactorDatabase
from quant_fund_agent.schemas import FactorRecord, FactorSource
from quant_fund_agent.workspace import Scope

log = logging.getLogger("simulation.factor_injection")


def resolve_prerun_db(name: str, config_name: str | None) -> Path:
    """Resolve a prerun reference to its researcher factor-DB path.

    Accepts a bare prerun name resolved under ``config_name``
    (``Scope(config_name, name)``), an explicit ``config/prerun`` pair, or a
    legacy flat prerun name (``data/factors/preruns/<name>/``).  Returns the path
    even if it does not exist (the caller reports the missing pool).
    """
    if "/" in name:
        cfg, _, pr = name.partition("/")
        return Scope(cfg, pr).factor_db_path
    if config_name:
        scoped = Scope(config_name, name).factor_db_path
        if scoped.exists():
            return scoped
    # Fall back to the legacy flat layout (compat shim handles older preruns).
    from quant_fund_agent.factors import preruns

    return preruns.db_path(name)


class PrerunFactorPool:
    """A shuffled, draw-without-replacement pool of prerun factor records.

    The union of the named preruns' RESEARCHER factors (de-duplicated by id,
    first occurrence wins) is shuffled once with a fixed seed, so a run is fully
    reproducible.  :meth:`draw` pops the next ``n`` records; once the pool is
    exhausted it simply returns fewer (and eventually none) — each factor enters
    the fund at most once.
    """

    def __init__(
        self,
        prerun_names: list[str],
        config_name: str | None = None,
        seed: int = 7,
    ) -> None:
        self.prerun_names = list(prerun_names)
        self.config_name = config_name
        records: dict[str, FactorRecord] = {}
        per_prerun: dict[str, int] = {}
        for name in self.prerun_names:
            db_path = resolve_prerun_db(name, config_name)
            if not db_path.exists():
                log.warning("inject pool: prerun %r has no factor DB at %s — skipped",
                            name, db_path)
                per_prerun[name] = 0
                continue
            db = FactorDatabase()
            db.load_from_json(db_path)
            facs = db.list_factors(source=FactorSource.RESEARCHER)
            per_prerun[name] = len(facs)
            for f in facs:
                records.setdefault(f.id, f)  # first prerun wins on id collision

        self._records = list(records.values())
        random.Random(seed).shuffle(self._records)
        self._cursor = 0
        self._drawn: list[str] = []
        log.info("inject pool: %d unique factors from %d prerun(s) %s",
                 len(self._records), len(self.prerun_names),
                 {k: per_prerun.get(k, 0) for k in self.prerun_names})

    # ── stats ────────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self._records)

    @property
    def remaining(self) -> int:
        return max(0, len(self._records) - self._cursor)

    @property
    def drawn_ids(self) -> list[str]:
        return list(self._drawn)

    @property
    def all_ids(self) -> list[str]:
        """Every factor id currently in the pool (drawn + undrawn)."""
        return [f.id for f in self._records]

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0

    # ── filtering ────────────────────────────────────────────────────────
    def keep_only(self, ids: set[str]) -> int:
        """Restrict the pool to factor ids in ``ids`` (call *before* drawing).

        Used to drop prerun factors that aren't computable on the live panel
        (e.g. they need a field this provider doesn't serve), so every injected
        factor is real.  Resets the draw cursor; returns the kept count.
        """
        self._records = [f for f in self._records if f.id in ids]
        self._cursor = 0
        self._drawn = []
        return len(self._records)

    # ── drawing ──────────────────────────────────────────────────────────
    def draw(self, n: int) -> list[FactorRecord]:
        """Pop the next ``n`` factor records (fewer near/at exhaustion)."""
        if n <= 0 or self.exhausted:
            return []
        end = min(self._cursor + n, len(self._records))
        out = self._records[self._cursor:end]
        self._cursor = end
        self._drawn.extend(f.id for f in out)
        return out
