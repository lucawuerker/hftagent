"""Autonomous, disk-bounded LOBSTER pull → convert → delete loop.

A multi-ticker, multi-year LOBSTER pull is far too large to download in one go
(~1 TB raw), but only a small scratch budget (<50 GB) is available.  This module
walks a **work list of chunks** (``ticker × time-window``) and, for each:

1. checks the **disk governor** (real free space + a soft budget) before fetching;
2. asks a :class:`Portal` to download + unzip the chunk's raw files into scratch;
3. converts them in place with :func:`...converter.convert_folder`
   (appending to ``ticker_data/{TICKER}/bin{YYYYMM}.csv``);
4. verifies the output, records progress in a **resumable JSON manifest**, then
5. **deletes the raw files** so the next chunk fits.

Everything is idempotent and restartable: re-running skips ``done`` chunks and
re-attempts ``failed``/partial ones.  ``--dry-run`` plans the chunks and prints
the disk math without touching the network — that path is exercised by tests.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from quant_fund_agent.data.lobster_ingest.converter import convert_folder

log = logging.getLogger("lobster_ingest.orchestrator")


# ── portal abstraction (Playwright driver lives in lobster_portal.py) ───────
class Portal(Protocol):
    """A source of raw LOBSTER day-files for one chunk.

    Implementations: :class:`...lobster_portal.PlaywrightLobsterPortal` (drives
    the web portal) and :class:`...lobster_portal.WatchFolderPortal` (manual
    download into a drop folder).  Both return the directory holding the
    unzipped ``*_message_*`` / ``*_orderbook_*`` CSVs.
    """

    def fetch(self, ticker: str, start: _dt.date, end: _dt.date, levels: int, dest: Path) -> Path:
        ...


# ── configuration ───────────────────────────────────────────────────────────
@dataclass
class IngestConfig:
    """Where to pull from, how to chunk it, and the disk budget."""

    tickers: list[str]
    start: str                              # ISO date, inclusive
    end: str                                # ISO date, inclusive
    levels: int = 5                         # LOB depth to request (sample on disk is 3)
    interval: int = 0                       # snapshot seconds; 0 = RAW message+orderbook
    chunk: str = "month"                    # "month" | "week"
    chunk_months: int = 1                    # when chunk == "month", months per chunk
    chunk_months_by_ticker: dict[str, int] = field(default_factory=dict)
    scratch_dir: str = "data/lobster_raw"   # raw download workspace (gitignored)
    out_dir: str = "ticker_data"            # converted output root
    manifest_path: str = "data/lobster_raw/manifest.json"

    # Disk governor (GB).  ``disk_budget_gb`` is a soft cap on scratch usage;
    # ``min_free_gb`` is a hard floor on the volume's real free space.
    disk_budget_gb: float = 45.0
    min_free_gb: float = 5.0
    # Size estimate per ticker per trading day.  The message file barely depends
    # on levels; the orderbook scales ~linearly with levels.  ``safety_factor``
    # pads for liquid names (e.g. AAPL ≫ GLD message rates).
    est_msg_gb_per_day: float = 0.08
    est_ob_gb_per_day_per_level: float = 0.034
    safety_factor: float = 2.0
    est_gb_per_month_by_ticker: dict[str, float] = field(default_factory=dict)

    # Portal polling (LOBSTER prepares data asynchronously).
    poll_interval_s: int = 60
    poll_timeout_s: int = 6 * 3600
    server_budget_gb: float = 80.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IngestConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def est_gb_per_day(self) -> float:
        return self.est_msg_gb_per_day + self.est_ob_gb_per_day_per_level * self.levels


# ── chunk model + manifest ──────────────────────────────────────────────────
@dataclass
class Chunk:
    ticker: str
    start: str          # ISO
    end: str            # ISO
    status: str = "pending"   # pending | placed | downloaded | converted | done | failed
    raw_dir: str | None = None
    out_files: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.ticker}_{self.start}_{self.end}"

    def est_gb(self, cfg: IngestConfig) -> float:
        days = int(np.busday_count(self.start, _next_day(self.end)))
        monthly = cfg.est_gb_per_month_by_ticker.get(self.ticker)
        if monthly is not None:
            return max(days, 1) / 21.0 * monthly
        return max(days, 1) * cfg.est_gb_per_day() * cfg.safety_factor


def _next_day(iso: str) -> str:
    return (_dt.date.fromisoformat(iso) + _dt.timedelta(days=1)).isoformat()


def plan_chunks(cfg: IngestConfig) -> list[Chunk]:
    """Expand ``tickers × [start, end]`` into per-ticker time-window chunks."""
    start, end = _dt.date.fromisoformat(cfg.start), _dt.date.fromisoformat(cfg.end)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    chunks: list[Chunk] = []
    for ticker in cfg.tickers:
        if cfg.chunk == "month":
            months = cfg.chunk_months_by_ticker.get(ticker, cfg.chunk_months)
            windows = _month_windows(start, end, months=max(1, int(months)))
        else:
            windows = _week_windows(start, end)
        chunks.extend(
            Chunk(ticker=ticker, start=a.isoformat(), end=b.isoformat())
            for a, b in windows
        )
    return chunks


def _month_windows(start: _dt.date, end: _dt.date, months: int = 1) -> list[tuple[_dt.date, _dt.date]]:
    out, cur = [], start
    while cur <= end:
        window_end = _add_months(cur, months) - _dt.timedelta(days=1)
        out.append((cur, min(window_end, end)))
        cur = window_end + _dt.timedelta(days=1)
    return out


def _add_months(date: _dt.date, months: int) -> _dt.date:
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    return _dt.date(year, month, 1)


def _week_windows(start: _dt.date, end: _dt.date) -> list[tuple[_dt.date, _dt.date]]:
    out, cur = [], start
    while cur <= end:
        wk_end = min(cur + _dt.timedelta(days=6), end)
        out.append((cur, wk_end))
        cur = wk_end + _dt.timedelta(days=1)
    return out


class Manifest:
    """Resumable per-chunk status, persisted as JSON next to the scratch dir."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.chunks: dict[str, Chunk] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.chunks = {k: Chunk(**v) for k, v in data.items()}

    def sync(self, planned: list[Chunk]) -> None:
        """Mirror the current plan without clobbering matching chunk statuses."""
        planned_keys = {ch.key for ch in planned}
        self.chunks = {k: v for k, v in self.chunks.items() if k in planned_keys}
        for ch in planned:
            self.chunks.setdefault(ch.key, ch)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: dataclasses.asdict(v) for k, v in self.chunks.items()}, indent=2)
        )

    def pending(self) -> list[Chunk]:
        return [c for c in self.chunks.values() if c.status not in ("done",)]


# ── disk governor ───────────────────────────────────────────────────────────
@dataclass
class DiskGovernor:
    scratch_dir: Path
    budget_gb: float
    min_free_gb: float

    def scratch_usage_gb(self) -> float:
        if not self.scratch_dir.exists():
            return 0.0
        total = sum(p.stat().st_size for p in self.scratch_dir.rglob("*") if p.is_file())
        return total / 1e9

    def free_gb(self) -> float:
        target = self.scratch_dir if self.scratch_dir.exists() else self.scratch_dir.parent
        return shutil.disk_usage(target).free / 1e9

    def check(self, need_gb: float) -> tuple[bool, str]:
        """Is it safe to download ``need_gb`` more into scratch right now?"""
        free, used = self.free_gb(), self.scratch_usage_gb()
        if free - need_gb < self.min_free_gb:
            return False, (
                f"only {free:.1f} GB free; downloading ~{need_gb:.1f} GB would drop "
                f"below the {self.min_free_gb:.0f} GB floor"
            )
        if used + need_gb > self.budget_gb:
            return False, (
                f"scratch already holds {used:.1f} GB; +{need_gb:.1f} GB would exceed "
                f"the {self.budget_gb:.0f} GB budget — convert/delete earlier chunks first"
            )
        return True, f"ok (free {free:.1f} GB, scratch {used:.1f} GB, need ~{need_gb:.1f} GB)"


# ── orchestrator ────────────────────────────────────────────────────────────
class Orchestrator:
    def __init__(self, cfg: IngestConfig, portal: Portal | None = None):
        self.cfg = cfg
        self.portal = portal
        self.scratch = Path(cfg.scratch_dir)
        self.manifest = Manifest(cfg.manifest_path)
        self.governor = DiskGovernor(self.scratch, cfg.disk_budget_gb, cfg.min_free_gb)

    # -- planning / reporting (no network) --
    def plan(self) -> list[Chunk]:
        planned = plan_chunks(self.cfg)
        self.manifest.sync(planned)
        return planned

    def dry_run(self) -> None:
        planned = self.plan()
        total_gb = sum(c.est_gb(self.cfg) for c in planned)
        max_chunk = max((c.est_gb(self.cfg) for c in planned), default=0.0)
        log.info(
            "planned %d chunks (%d tickers × %d windows, %s granularity, level %d)",
            len(planned), len(self.cfg.tickers),
            len(planned) // max(len(self.cfg.tickers), 1), self.cfg.chunk, self.cfg.levels,
        )
        log.info(
            "est. total raw moved ≈ %.0f GB; largest single chunk ≈ %.1f GB "
            "(processed one-at-a-time, deleted after convert)", total_gb, max_chunk,
        )
        log.info(
            "disk: budget %.0f GB, min-free %.0f GB, current free %.1f GB, scratch %.1f GB",
            self.cfg.disk_budget_gb, self.cfg.min_free_gb,
            self.governor.free_gb(), self.governor.scratch_usage_gb(),
        )
        ok, msg = self.governor.check(max_chunk)
        log.info("largest-chunk pre-flight: %s — %s", "OK" if ok else "BLOCKED", msg)
        if max_chunk > self.cfg.disk_budget_gb:
            log.warning(
                "largest chunk (~%.1f GB) exceeds the budget; use chunk='week' or raise "
                "disk_budget_gb", max_chunk,
            )
        self.manifest.save()

    # -- the real loop --
    def run(self) -> None:
        if self.portal is None:
            raise RuntimeError("a Portal is required for run(); use dry_run() to plan only")
        self.plan()
        for ch in self.manifest.pending():
            try:
                self._process(ch)
            except Exception as e:  # one bad chunk must not abort the whole pull
                ch.status, ch.error = "failed", f"{type(e).__name__}: {e}"
                log.exception("chunk %s failed", ch.key)
            self.manifest.save()
        done = sum(1 for c in self.manifest.chunks.values() if c.status == "done")
        log.info("ingest finished: %d/%d chunks done", done, len(self.manifest.chunks))

    def _process(self, ch: Chunk) -> None:
        cfg = self.cfg
        if ch.status in ("pending", "failed"):
            need = ch.est_gb(cfg)
            ok, msg = self.governor.check(need)
            if not ok:
                raise RuntimeError(f"disk governor blocked {ch.key}: {msg}")
            log.info("downloading %s (%s)", ch.key, msg)
            dest = self.scratch / ch.key
            raw_dir = self.portal.fetch(
                ch.ticker,
                _dt.date.fromisoformat(ch.start),
                _dt.date.fromisoformat(ch.end),
                cfg.levels,
                dest,
            )
            ch.raw_dir, ch.status = str(raw_dir), "downloaded"
            self.manifest.save()

        if ch.status == "downloaded":
            log.info("converting %s", ch.key)
            summary = convert_folder(ch.raw_dir, ticker=ch.ticker, out_dir=cfg.out_dir)
            if summary.days == 0:
                raise RuntimeError(f"no day-pairs found in {ch.raw_dir}")
            ch.out_files = [str(p) for p in summary.out_files]
            ch.status = "converted"
            self.manifest.save()

        if ch.status == "converted":
            self._delete_raw(ch)
            ch.status = "done"
            log.info("done %s → %s (raw freed)", ch.key, ", ".join(ch.out_files))

    def _delete_raw(self, ch: Chunk) -> None:
        if ch.raw_dir and Path(ch.raw_dir).exists():
            shutil.rmtree(ch.raw_dir, ignore_errors=True)
        ch.raw_dir = None

    # ── live LOBSTER flow (request → ingest finished orders) ──
    def place_raw(self, portal) -> None:
        """Submit one RAW ``requestdata.php`` request per planned chunk.

        The single-form product (message+orderbook) is the only one with the
        event data needed for the flow/trade columns; the bulk form is book-only.
        """
        planned = self.plan()
        self.manifest.save()
        placed = 0
        with portal.session() as page:
            for ch in planned:
                current = self.manifest.chunks[ch.key]
                if current.status not in ("pending", "failed"):
                    continue
                portal.place_single_request(
                    page,
                    current.ticker,
                    _dt.date.fromisoformat(current.start),
                    _dt.date.fromisoformat(current.end),
                    self.cfg.levels,
                )
                current.status = "placed"
                current.error = None
                placed += 1
                self.manifest.save()
        log.info("placed %d raw chunk request(s); collect with --ingest or --run-live",
                 placed)

    def place_bulk(self, portal, interval: int | None = None) -> None:
        """Submit ONE bulk request for all configured tickers/dates/level.

        ``interval=0`` (the default) requests the **raw** message+orderbook
        product (full 19 columns); a positive interval requests the sampled book.
        Data is prepared asynchronously and later collected by :meth:`ingest_ready`.
        """
        iv = self.cfg.interval if interval is None else interval
        with portal.session() as page:
            portal.place_bulk_request(
                page,
                list(self.cfg.tickers),
                _dt.date.fromisoformat(self.cfg.start),
                _dt.date.fromisoformat(self.cfg.end),
                self.cfg.levels,
                iv,
            )

    def ingest_ready(self, portal, only_raw: bool = False) -> None:
        """Download + stream-convert + delete every finished order on mydata.php.

        Idempotent: a tiny ``orders_done.json`` records ingested order keys so
        re-runs skip them. Each ``.7z`` is converted day-by-day
        (:func:`converter.convert_archive`) then deleted to respect the disk budget.
        """
        from quant_fund_agent.data.lobster_ingest.converter import convert_archive

        self.scratch.mkdir(parents=True, exist_ok=True)
        done = self._load_done()
        ingested = 0
        with portal.session() as page:
            orders = portal.list_ready_orders(page)
            log.info("mydata.php lists %d ready order(s)", len(orders))
            for o in orders:
                tag = f"{o.ticker} {o.start}->{o.end} L{o.level} I{o.interval}"
                if o.key in done:
                    log.info("skip %s (already ingested)", tag)
                    continue
                if only_raw and not o.is_raw:
                    log.info("skip %s (sampled; --only-raw set)", tag)
                    continue
                free = self.governor.free_gb()
                if free < self.cfg.min_free_gb + 2:
                    raise RuntimeError(f"low disk ({free:.1f} GB free) before {tag}")
                if not o.is_raw:
                    log.warning("%s is SAMPLED (interval %d) — only book columns "
                                "will be populated", tag, o.interval)
                arc = portal.download_order(page, o, self.scratch / "orders")
                try:
                    summary = convert_archive(arc, ticker=o.ticker, out_dir=self.cfg.out_dir)
                    log.info("converted %s: %d days, %d rows → %s",
                             o.ticker, summary.days, summary.rows,
                             ", ".join(p.name for p in summary.out_files[:3]) + " …")
                finally:
                    arc.unlink(missing_ok=True)  # free the archive before the next order
                if hasattr(portal, "delete_order"):
                    portal.delete_order(page, o)
                done.add(o.key)
                self._save_done(done)
                ingested += 1
        log.info("ingest_ready finished: %d new order(s) ingested (%d total)", ingested, len(done))

    def run_live(self, portal) -> None:
        """Continuously request, ingest, and delete planned RAW chunks.

        This keeps LOBSTER server storage bounded by submitting only enough
        asynchronous requests to fill ``server_budget_gb`` approximately, then
        ingesting/deleting ready chunks before placing more.
        """
        from quant_fund_agent.data.lobster_ingest.converter import convert_archive

        self.scratch.mkdir(parents=True, exist_ok=True)
        self.plan()
        self.manifest.save()
        deadline = time.time() + self.cfg.poll_timeout_s

        with portal.session() as page:
            while True:
                remaining = [c for c in self.manifest.chunks.values() if c.status != "done"]
                if not remaining:
                    log.info("run_live finished: all %d chunks done", len(self.manifest.chunks))
                    return

                orders = portal.list_ready_orders(page)
                did_work = False
                for ch in list(remaining):
                    order = self._matching_order(ch, orders)
                    if order is None:
                        continue
                    free = self.governor.free_gb()
                    if free < self.cfg.min_free_gb + 2:
                        raise RuntimeError(f"low disk ({free:.1f} GB free) before {ch.key}")
                    log.info("ingesting ready chunk %s via order %s", ch.key, order.order_id)
                    arc = portal.download_order(page, order, self.scratch / "orders")
                    try:
                        summary = convert_archive(arc, ticker=order.ticker, out_dir=self.cfg.out_dir)
                        ch.out_files = [str(p) for p in summary.out_files]
                        log.info("converted %s: %d days, %d rows", ch.key, summary.days, summary.rows)
                    finally:
                        arc.unlink(missing_ok=True)
                    if hasattr(portal, "delete_order"):
                        portal.delete_order(page, order)
                    ch.status = "done"
                    ch.raw_dir = None
                    ch.error = None
                    self.manifest.save()
                    did_work = True

                in_flight = [
                    c for c in self.manifest.chunks.values()
                    if c.status == "placed" and self._matching_order(c, orders) is None
                ]
                in_flight_gb = sum(c.est_gb(self.cfg) for c in in_flight)
                for ch in self.manifest.chunks.values():
                    if ch.status not in ("pending", "failed"):
                        continue
                    need = ch.est_gb(self.cfg)
                    if in_flight_gb + need > self.cfg.server_budget_gb:
                        continue
                    log.info("placing RAW chunk %s (est server %.1f GB; in flight %.1f/%.1f GB)",
                             ch.key, need, in_flight_gb, self.cfg.server_budget_gb)
                    portal.place_single_request(
                        page,
                        ch.ticker,
                        _dt.date.fromisoformat(ch.start),
                        _dt.date.fromisoformat(ch.end),
                        self.cfg.levels,
                    )
                    ch.status = "placed"
                    ch.error = None
                    in_flight_gb += need
                    did_work = True
                    self.manifest.save()

                if did_work:
                    deadline = time.time() + self.cfg.poll_timeout_s
                    continue
                if time.time() > deadline:
                    raise TimeoutError("no LOBSTER chunk became ready before poll_timeout_s")
                log.info("waiting %ds for LOBSTER; %d chunk(s) remaining, %.1f GB estimated in flight",
                         self.cfg.poll_interval_s, len(remaining), in_flight_gb)
                time.sleep(self.cfg.poll_interval_s)

    def _matching_order(self, ch: Chunk, orders) -> object | None:
        for o in orders:
            if (
                o.is_raw
                and o.ticker == ch.ticker
                and o.start == ch.start
                and o.end == ch.end
                and o.level == self.cfg.levels
            ):
                return o
        return None

    def _done_path(self) -> Path:
        return self.scratch / "orders_done.json"

    def _load_done(self) -> set[str]:
        p = self._done_path()
        return set(json.loads(p.read_text())) if p.exists() else set()

    def _save_done(self, done: set[str]) -> None:
        self._done_path().parent.mkdir(parents=True, exist_ok=True)
        self._done_path().write_text(json.dumps(sorted(done), indent=2))
