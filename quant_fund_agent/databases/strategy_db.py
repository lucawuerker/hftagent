"""In-memory strategy database.

Stores ``StrategyRecord`` metadata that links to concrete
``BaseStrategy`` subclasses via the ``class_name`` field.

The database also persists per-strategy return series under
``<returns_dir>/<strategy_id>.csv`` and caches the pairwise strategy
correlation matrix, both of which the Portfolio Manager Agent reads
directly.

Caching model
-------------
The PM agent typically runs many times over the course of a long
backtest while the underlying strategy book changes much less often.
We therefore *don't* recompute correlations on every PM invocation;
instead we maintain three layers of cache:

1. ``_returns_cache``       – in-memory copy of every per-strategy PnL
   series.  Populated lazily as ``load_returns`` / ``save_returns`` are
   called; each CSV is read from disk at most once per process.
2. ``_correlation_matrix``  – cached pairwise correlations.
3. ``_covariance_matrix``   – cached bar-scale covariance.  The
   covariance accessor multiplies by ``annualisation_factor`` on access
   so caching is independent of the units the caller asks for.

A single ``_correlations_dirty`` flag invalidates (2) and (3) together.
Mutations that affect the matrices flip the flag::

    add_strategy / register_strategy / save_returns /
    append_returns / remove_strategy / load_from_json   → set dirty

Operations that only touch metadata DO NOT::

    set_pm_status / flag_strategy / retire_strategy     → no-op

The first ``correlation_matrix`` / ``covariance_matrix`` read after a
mutation triggers exactly one recompute; subsequent reads are O(1).
This collapses the per-PM-call cost from O(N² · T + #strategies disk
reads) to O(1) in the steady state, which matters in a year-long
backtest with daily (or finer) PM rebalances.

Typical lifecycle::

    db = StrategyDatabase()
    db.load_from_json("data/strategies/strategy_db.json")

    # Add a new strategy AFTER the statistician has approved it.
    # This writes the record + persists the per-bar returns to disk +
    # marks the correlation cache dirty.  The next PM read will trigger
    # exactly one recompute across the full book.
    db.register_strategy(record, portfolio_returns=oos_returns)

    # Inside a backtest loop, extend a live strategy's PnL one bar at a
    # time without touching disk on every bar:
    db.append_returns(strategy_id, new_bars)

    # Used by the PM agent (lazy, cache-aware):
    cov  = db.covariance_matrix(annualisation_factor=BARS_PER_YEAR)
    corr = db.correlation_matrix
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from quant_fund_agent.schemas import (
    PMStatus,
    StrategyRecord,
    StrategyStatus,
)

log = logging.getLogger("strategy_db")


DEFAULT_RETURNS_DIR = Path("data/strategies/returns")


class StrategyDatabase:
    """Registry of trading strategies + their PnL series + correlations."""

    def __init__(self, returns_dir: str | Path | None = None) -> None:
        # Resolve the returns dir live from ``STRATEGY_RETURNS_DIR`` when no
        # explicit dir is given, so a scoped run (run_fund/simulator) keeps its
        # per-strategy PnL CSVs under its own workspace instead of colliding in
        # the global ``data/strategies/returns``.  An explicit arg always wins
        # (the simulator passes ``config.returns_dir``).
        if returns_dir is None:
            returns_dir = os.getenv("STRATEGY_RETURNS_DIR", str(DEFAULT_RETURNS_DIR))
        self._strategies: dict[str, StrategyRecord] = {}
        self._returns_dir = Path(returns_dir)

        # In-memory caches.  See module docstring for invariants.
        self._returns_cache: dict[str, pd.Series] = {}
        self._correlation_matrix: pd.DataFrame = pd.DataFrame()
        self._covariance_matrix_bar: pd.DataFrame = pd.DataFrame()
        self._correlations_dirty: bool = True

        # Stats for observability — exposed via ``cache_stats``.
        self._n_recomputes: int = 0

    # ── CRUD ──────────────────────────────────────────────────────────

    def add_strategy(self, strategy: StrategyRecord) -> None:
        self._strategies[strategy.id] = strategy
        self._mark_dirty()

    def get_strategy(self, strategy_id: str) -> StrategyRecord | None:
        return self._strategies.get(strategy_id)

    def list_strategies(
        self,
        status: StrategyStatus | None = None,
        pm_status: PMStatus | None = None,
    ) -> list[StrategyRecord]:
        out = list(self._strategies.values())
        if status is not None:
            out = [s for s in out if s.status == status]
        if pm_status is not None:
            out = [s for s in out if s.pm_status == pm_status]
        return out

    def update_strategy(self, strategy: StrategyRecord) -> None:
        # We can't tell from here whether the update touched anything
        # correlation-relevant, so be conservative and invalidate.  Most
        # in-place PM-status mutations go through ``set_pm_status``
        # instead and won't hit this path.
        self._strategies[strategy.id] = strategy
        self._mark_dirty()

    def remove_strategy(self, strategy_id: str) -> None:
        self._strategies.pop(strategy_id, None)
        self._returns_cache.pop(strategy_id, None)
        # Best-effort removal of persisted returns file.
        ret_path = self._returns_path(strategy_id)
        if ret_path.exists():
            try:
                ret_path.unlink()
            except OSError:  # pragma: no cover
                pass
        self._mark_dirty()

    # ── PM-status helpers (do NOT affect correlations) ────────────────

    def set_pm_status(
        self,
        strategy_id: str,
        pm_status: PMStatus,
        notes: str = "",
    ) -> None:
        rec = self._strategies.get(strategy_id)
        if rec is None:
            log.warning("set_pm_status: unknown strategy_id=%s", strategy_id)
            return
        rec.pm_status = pm_status
        if notes:
            rec.pm_notes = (rec.pm_notes + "\n" + notes).strip() if rec.pm_notes else notes

    def flag_strategy(self, strategy_id: str, reason: str) -> None:
        rec = self._strategies.get(strategy_id)
        if rec is None:
            return
        rec.pm_status = PMStatus.FLAGGED_FOR_REVIEW
        rec.flagged_reason = reason

    def retire_strategy(self, strategy_id: str, reason: str = "") -> None:
        rec = self._strategies.get(strategy_id)
        if rec is None:
            return
        rec.pm_status = PMStatus.RETIRED
        if reason:
            rec.flagged_reason = reason

    # ── cache invariants ──────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        """Invalidate the cached correlation / covariance matrices."""
        self._correlations_dirty = True

    def is_correlation_cache_dirty(self) -> bool:
        return self._correlations_dirty

    def cache_stats(self) -> dict[str, int | bool]:
        """For introspection / tests.  Cheap, side-effect free."""
        return {
            "dirty": self._correlations_dirty,
            "n_strategies_cached": len(self._returns_cache),
            "n_recomputes": self._n_recomputes,
            "matrix_size": int(self._correlation_matrix.shape[0]),
        }

    # ── returns-series persistence ────────────────────────────────────

    def _returns_path(self, strategy_id: str) -> Path:
        return self._returns_dir / f"{strategy_id}.csv"

    def save_returns(self, strategy_id: str, returns: pd.Series) -> Path:
        """Persist a strategy's per-bar PnL series to disk + cache it."""
        if strategy_id not in self._strategies:
            raise KeyError(f"Unknown strategy_id: {strategy_id}")
        if returns is None or len(returns) == 0:
            log.warning("save_returns: empty series for %s — skipping.", strategy_id)
            return self._returns_path(strategy_id)

        path = self._returns_path(strategy_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = returns.copy()
        s.index = pd.to_datetime(s.index)
        s.name = "portfolio_return"
        s.to_csv(path, index_label="timestamp")
        # Refresh the in-memory cache so we don't re-read what we just wrote.
        cached = s.copy()
        cached.name = strategy_id
        self._returns_cache[strategy_id] = cached
        # Backfill record with path so it survives a save/load round-trip.
        rec = self._strategies[strategy_id]
        rec.returns_path = str(path)
        self._mark_dirty()
        return path

    def append_returns(
        self,
        strategy_id: str,
        new_bars: pd.Series,
        persist_every_n: int | None = None,
    ) -> None:
        """Append new per-bar PnL to a strategy without rewriting the whole CSV.

        Designed for the backtest/simulation loop where each strategy's
        PnL grows one (or a few) bars at a time.  The in-memory cache is
        updated on every call (cheap); the on-disk CSV is rewritten only
        every ``persist_every_n`` calls (None means every time).

        We *append* to the disk file with ``mode='a'`` and no header
        when possible, but if the cache has been mutated in memory we
        rewrite the whole file to keep the two in sync.
        """
        if strategy_id not in self._strategies:
            raise KeyError(f"Unknown strategy_id: {strategy_id}")
        if new_bars is None or len(new_bars) == 0:
            return

        new_bars = new_bars.copy()
        new_bars.index = pd.to_datetime(new_bars.index)
        new_bars.name = strategy_id

        existing = self._returns_cache.get(strategy_id)
        if existing is None:
            # Lazy-load from disk if there is one, then concatenate.
            on_disk = self._load_returns_from_disk(strategy_id)
            existing = on_disk if on_disk is not None else pd.Series(dtype=float, name=strategy_id)

        # Drop overlap (idempotent appends).
        combined = pd.concat([existing, new_bars])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        self._returns_cache[strategy_id] = combined

        # Persistence — opportunistic.  For very long sims a caller may
        # choose to flush periodically.
        if persist_every_n is None or len(combined) % max(persist_every_n, 1) == 0:
            path = self._returns_path(strategy_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            to_write = combined.copy()
            to_write.name = "portfolio_return"
            to_write.to_csv(path, index_label="timestamp")
            self._strategies[strategy_id].returns_path = str(path)

        self._mark_dirty()

    def flush_returns(self, strategy_id: str | None = None) -> None:
        """Force a CSV flush of one (or all) strategies' cached returns.

        Useful at the end of a simulation if you've been appending with
        ``persist_every_n>1`` and want everything on disk before exit.
        """
        ids = [strategy_id] if strategy_id is not None else list(self._returns_cache.keys())
        for sid in ids:
            cached = self._returns_cache.get(sid)
            if cached is None or sid not in self._strategies:
                continue
            path = self._returns_path(sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            to_write = cached.copy()
            to_write.name = "portfolio_return"
            to_write.to_csv(path, index_label="timestamp")
            self._strategies[sid].returns_path = str(path)

    def _load_returns_from_disk(self, strategy_id: str) -> pd.Series | None:
        rec = self._strategies.get(strategy_id)
        if rec is None:
            return None
        path = Path(rec.returns_path) if rec.returns_path else self._returns_path(strategy_id)
        if not path.exists():
            return None
        df = pd.read_csv(path, parse_dates=["timestamp"])
        s = df.set_index("timestamp")["portfolio_return"]
        s.name = strategy_id
        return s

    def load_returns(self, strategy_id: str) -> pd.Series | None:
        """Return a strategy's PnL series.  Memoised after first call."""
        if strategy_id in self._returns_cache:
            return self._returns_cache[strategy_id]
        s = self._load_returns_from_disk(strategy_id)
        if s is not None:
            self._returns_cache[strategy_id] = s
        return s

    def get_all_returns(
        self, strategy_ids: list[str] | None = None
    ) -> dict[str, pd.Series]:
        """Load every available returns series (or just the requested ones).

        Uses the in-memory cache; only reads disk for IDs not yet seen.
        """
        ids = strategy_ids or list(self._strategies.keys())
        out: dict[str, pd.Series] = {}
        for sid in ids:
            s = self.load_returns(sid)
            if s is not None and len(s) > 1:
                out[sid] = s
        return out

    # ── correlation / covariance (lazy + cached) ──────────────────────

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Pairwise Pearson correlation, recomputed lazily on dirty cache."""
        if self._correlations_dirty:
            self._recompute_matrices()
        return self._correlation_matrix

    def covariance_matrix(
        self,
        annualisation_factor: float = 1.0,
        strategy_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        """Empirical covariance, optionally annualised and/or restricted.

        Returns a view on the cached bar-scale covariance multiplied by
        ``annualisation_factor``.  Falls back to a vol×ρ×vol expansion
        when no returns are available for some strategies.
        """
        if self._correlations_dirty:
            self._recompute_matrices()

        if not self._covariance_matrix_bar.empty:
            cov = self._covariance_matrix_bar.copy()
            if annualisation_factor != 1.0:
                cov = cov * annualisation_factor
            if strategy_ids is not None:
                cov = cov.reindex(index=strategy_ids, columns=strategy_ids)
            return cov

        # No returns available — synthesise from summary stats on the
        # strategy records (vol×ρ×vol expansion).
        from quant_fund_agent.portfolio.correlations import (
            covariance_from_vols_and_corr,
        )
        ids = strategy_ids or list(self._strategies.keys())
        vols: dict[str, float] = {}
        for sid in ids:
            rec = self._strategies.get(sid)
            if rec is None or rec.backtest_metrics is None:
                continue
            v = rec.backtest_metrics.annualised_volatility
            if v is None:
                continue
            vols[sid] = float(v)
        if not vols:
            return pd.DataFrame()
        return covariance_from_vols_and_corr(vols, self._correlation_matrix)

    def _recompute_matrices(self, min_overlap: int = 30) -> None:
        """Single source of truth for the cached corr/cov matrices.

        Reads from ``_returns_cache`` (which is itself lazy-populated
        from disk), recomputes the bar-scale corr & cov, and writes per-
        strategy correlation dicts back onto each ``StrategyRecord``.
        """
        from quant_fund_agent.portfolio.correlations import (
            compute_correlation_matrix,
            compute_covariance_matrix,
        )

        returns = self.get_all_returns()
        if len(returns) >= 2:
            corr = compute_correlation_matrix(returns, min_overlap=min_overlap)
            cov = compute_covariance_matrix(returns, min_overlap=min_overlap)
        elif len(returns) == 1:
            sid = next(iter(returns.keys()))
            corr = pd.DataFrame([[1.0]], index=[sid], columns=[sid])
            var = float(returns[sid].var(ddof=1))
            cov = pd.DataFrame([[var]], index=[sid], columns=[sid])
        else:
            corr = pd.DataFrame()
            cov = pd.DataFrame()

        self._correlation_matrix = corr
        self._covariance_matrix_bar = cov

        # Backfill per-record correlation dicts so they survive a JSON
        # save/load round-trip.  Only entries for known strategies kept.
        all_ids = set(self._strategies.keys())
        for sid, rec in self._strategies.items():
            if sid not in corr.index:
                rec.correlations = {}
                continue
            row = corr.loc[sid].drop(labels=[sid], errors="ignore")
            rec.correlations = {
                k: float(v) for k, v in row.dropna().items() if k in all_ids
            }

        self._correlations_dirty = False
        self._n_recomputes += 1
        log.debug(
            "Correlation cache recomputed: %d × %d (recompute #%d)",
            corr.shape[0], corr.shape[1], self._n_recomputes,
        )

    def refresh_correlations(self, min_overlap: int = 30) -> pd.DataFrame:
        """Force a recomputation of the correlation / covariance cache.

        Most callers should just read ``correlation_matrix`` and let the
        dirty-flag mechanism trigger recomputation automatically.  This
        method exists for explicit "I just changed the on-disk files
        behind the DB's back" scenarios.
        """
        self._mark_dirty()
        # Drop the in-memory cache too, in case files changed on disk.
        self._returns_cache.clear()
        self._recompute_matrices(min_overlap=min_overlap)
        return self._correlation_matrix

    # ── one-call registration: record + returns + dirty flag ──

    def register_strategy(
        self,
        strategy: StrategyRecord,
        portfolio_returns: pd.Series | None = None,
    ) -> None:
        """Add (or upsert) a strategy and mark the correlation cache dirty.

        This is the canonical entry point the architect/statistician
        pipeline should call once a strategy is approved.  It does NOT
        eagerly recompute correlations — the next PM read will do that
        once.  In a backtest where strategies are added in clusters,
        this batches the work automatically.
        """
        self.add_strategy(strategy)  # marks dirty
        if portfolio_returns is not None and len(portfolio_returns) > 0:
            self.save_returns(strategy.id, portfolio_returns)  # marks dirty

    # ── persistence ───────────────────────────────────────────────────

    def save_to_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = [s.model_dump(mode="json") for s in self._strategies.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))

    def load_from_json(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        for raw in json.loads(path.read_text()):
            strategy = StrategyRecord.model_validate(raw)
            self._strategies[strategy.id] = strategy
        # Don't eagerly recompute — let the first PM read trigger it.
        self._mark_dirty()
        self._returns_cache.clear()
