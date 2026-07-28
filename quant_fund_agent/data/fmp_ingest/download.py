"""Orchestration: turn a point-in-time universe into a complete local archive.

One **unit of work** is a ``(endpoint, period, symbol)`` triple that lands as one
parquet file and one manifest row.  Units for the same ticker are executed
together on one worker thread, because the *first* one — the adjusted price
series — is what resolves the ticker to a symbol FMP will serve, and every later
call reuses that resolution.

Concurrency is worker threads over **symbols**, throttled by a single shared
calls-per-minute budget (:class:`~.client.RateLimiter`).  That keeps the plan
limit respected globally no matter how many workers are running, and makes the
wall-clock a simple function of total calls ÷ rate.

Nothing here raises for a per-symbol failure: a run over ~1 300 tickers must
survive bad symbols, transient 5xx and plan gates, record them, and continue.
Files are written under the **membership ticker** (not the resolved vendor
symbol), so the panel layer can look a name up by the ticker it carried while it
was an index member.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

from quant_fund_agent.data.fmp_ingest.capabilities import Capabilities
from quant_fund_agent.data.fmp_ingest.client import FMPClient
from quant_fund_agent.data.fmp_ingest.endpoints import (
    Endpoint,
    endpoints_for_groups,
    work_units,
)
from quant_fund_agent.data.fmp_ingest.store import (
    STATUS_EMPTY,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_RESTRICTED,
    Archive,
    ManifestEntry,
)
from quant_fund_agent.data.fmp_ingest.symbols import (
    Resolution,
    candidate_symbols,
    spell_coverage,
    spell_windows,
)

log = logging.getLogger("data.fmp_ingest.download")

#: Fallback chunk length for windowed endpoints when the probe says a single
#: request does not reach the start of the window.
DEFAULT_WINDOW_YEARS = 5

#: Coarse per-endpoint payload sizes (MB per symbol over ~22 years), used only
#: for the dry-run bandwidth estimate.  Measured off the live probe responses.
_MB_PER_SYMBOL: dict[str, float] = {
    "prices_adjusted": 0.75, "prices_unadjusted": 1.10,
    "dividends": 0.02, "splits": 0.002,
    "income_statement": 0.12, "balance_sheet": 0.18, "cash_flow": 0.14,
    "ratios": 0.20, "key_metrics": 0.15, "financial_growth": 0.14,
    "enterprise_values": 0.03, "earnings": 0.03,
    "profile": 0.004, "market_cap": 0.40, "shares_float": 0.001,
}


@dataclass
class DownloadConfig:
    start: str = "2004-01-01"
    end: str = ""
    groups: tuple[str, ...] = ("prices", "fundamentals", "reference")
    indices: tuple[str, ...] = ("sp500", "nasdaq100")
    symbols: tuple[str, ...] = ()      # explicit override of the PIT universe
    max_symbols: int | None = None     # cap for a smoke run
    workers: int = 8
    retry_errors: bool = True          # re-attempt units that errored last run
    window_years: int | None = None    # override the probe's chunking decision

    def __post_init__(self) -> None:
        if not self.end:
            self.end = date.today().isoformat()


@dataclass
class DownloadReport:
    started_at: str = ""
    finished_at: str = ""
    symbols: int = 0
    units_total: int = 0
    units_done: int = 0
    units_skipped: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    resolutions: list[Resolution] = field(default_factory=list)
    client_stats: dict = field(default_factory=dict)
    archive_mb: float = 0.0

    def bump(self, status: str) -> None:
        self.by_status[status] = self.by_status.get(status, 0) + 1


class Downloader:
    """Drives the whole pull.  Construct, then call :meth:`run`."""

    def __init__(
        self,
        client: FMPClient,
        archive: Archive,
        config: DownloadConfig,
        *,
        capabilities: Capabilities | None = None,
        spells: pd.DataFrame | None = None,
        renames: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.archive = archive
        self.config = config
        self.caps = capabilities or Capabilities()
        self.spells = spells
        self.renames = renames or {}
        self.report = DownloadReport()
        self._start_ts = pd.Timestamp(config.start)
        self._end_ts = pd.Timestamp(config.end)

    # ── planning ────────────────────────────────────────────────────────────

    def endpoints(self) -> list[Endpoint]:
        eps = endpoints_for_groups(self.config.groups, kind="symbol")
        return [e for e in eps if self.caps.allows(e)]

    def _chunks(self, endpoint: Endpoint) -> list[tuple[str, str]]:
        """Date windows for a windowed endpoint (a single window when allowed)."""
        if not endpoint.windowed:
            return []
        years = self.config.window_years or endpoint.window_years
        if years is None:
            years = None if self.caps.price_full_range_ok else DEFAULT_WINDOW_YEARS
        if years is None:
            return [(self.config.start, self.config.end)]
        out: list[tuple[str, str]] = []
        cursor = self._start_ts
        step = pd.DateOffset(years=int(years))
        while cursor < self._end_ts:
            stop = min(cursor + step - pd.Timedelta(days=1), self._end_ts)
            out.append((cursor.date().isoformat(), stop.date().isoformat()))
            cursor = stop + pd.Timedelta(days=1)
        return out

    def plan(self, symbols: list[str]) -> dict:
        """Dry-run estimate: units, HTTP calls, bandwidth, wall clock."""
        eps = self.endpoints()
        units = calls = 0
        mb = 0.0
        per_endpoint: dict[str, dict] = {}
        for ep in eps:
            n_periods = len(work_units(ep))
            n_chunks = max(1, len(self._chunks(ep)))
            ep_units = n_periods * len(symbols)
            ep_calls = ep_units * n_chunks
            units += ep_units
            calls += ep_calls
            ep_mb = _MB_PER_SYMBOL.get(ep.name, 0.05) * len(symbols) * n_periods
            mb += ep_mb
            per_endpoint[ep.name] = {
                "units": ep_units, "calls": ep_calls,
                "periods": n_periods, "chunks": n_chunks, "mb": round(ep_mb, 1),
            }
        rate = self.client.limiter.per_minute
        return {
            "symbols": len(symbols),
            "endpoints": len(eps),
            "units": units,
            "calls": calls,
            "estimated_mb": round(mb, 1),
            "estimated_gb": round(mb / 1000, 2),
            "rate_per_minute": rate,
            # 2 dp so a small smoke plan doesn't display a misleading "0.0 min"
            "estimated_minutes": round(calls / max(rate, 1), 2),
            "per_endpoint": per_endpoint,
        }

    # ── universe ────────────────────────────────────────────────────────────

    def universe(self) -> list[str]:
        """Point-in-time union across the configured indices."""
        if self.config.symbols:
            symbols = [s.strip().upper() for s in self.config.symbols if s.strip()]
        else:
            from quant_fund_agent.data.membership import union_members

            seen: list[str] = []
            for index in self.config.indices:
                for t in union_members(self.config.start, self.config.end, index=index):
                    if t not in seen:
                        seen.append(t)
            symbols = sorted(seen)
        if self.config.max_symbols:
            symbols = symbols[: self.config.max_symbols]
        return symbols

    # ── fetching ────────────────────────────────────────────────────────────

    def fetch_global(self, groups: tuple[str, ...] = ("index",)) -> list[ManifestEntry]:
        """Fetch the run-level endpoints (index constituents, delisted list …)."""
        out: list[ManifestEntry] = []
        for ep in endpoints_for_groups(groups, kind="global"):
            entry = self._fetch_global_endpoint(ep)
            self.archive.record(entry)
            self.report.bump(entry.status)
            out.append(entry)
            log.info("global %-34s %-10s rows=%d", ep.name, entry.status, entry.rows)
        return out

    def _fetch_global_endpoint(self, ep: Endpoint) -> ManifestEntry:
        params: dict = dict(ep.params)
        if ep.limit is not None:
            params["limit"] = self.caps.limit_for(ep)
        rows: list[dict] = []
        calls = n_bytes = 0
        if ep.paginated:
            for page in range(200):
                res = self.client.get(ep.url(), {**params, "page": page})
                calls += 1
                n_bytes += res.n_bytes
                if res.restricted:
                    return ManifestEntry(ep.key(), ep.name, status=STATUS_RESTRICTED,
                                         calls=calls, n_bytes=n_bytes, error=res.error)
                if not res.ok or not res.rows:
                    break
                rows.extend(res.rows)
                if len(res.rows) < (params.get("limit") or 100):
                    break
        else:
            res = self.client.get(ep.url(), params)
            calls, n_bytes = 1, res.n_bytes
            if res.restricted:
                return ManifestEntry(ep.key(), ep.name, status=STATUS_RESTRICTED,
                                     calls=calls, n_bytes=n_bytes, error=res.error)
            if not res.ok:
                return ManifestEntry(ep.key(), ep.name, status=STATUS_ERROR,
                                     calls=calls, n_bytes=n_bytes, error=res.error)
            rows = res.rows
        n_rows, first, last = self.archive.write(ep, rows)
        return ManifestEntry(
            ep.key(), ep.name, status=STATUS_OK if n_rows else STATUS_EMPTY,
            rows=n_rows, calls=calls, n_bytes=n_bytes, first_date=first, last_date=last,
        )

    def _fetch_symbol_endpoint(
        self, ep: Endpoint, ticker: str, vendor_symbol: str, period: str | None
    ) -> ManifestEntry:
        """One ``(endpoint, period, symbol)`` unit — possibly several requests."""
        key = ep.key(ticker, period)
        base: dict = {"symbol": vendor_symbol, **ep.params}
        if period:
            base["period"] = period
        limit = self.caps.limit_for(ep)
        if limit is not None:
            base["limit"] = limit

        windows = self._chunks(ep) or [None]
        rows: list[dict] = []
        calls = n_bytes = 0
        first_error: str | None = None
        restricted = False
        for window in windows:
            params = dict(base)
            if window is not None:
                params["from"], params["to"] = window
            res = self.client.get(ep.url(), params)
            calls += 1
            n_bytes += res.n_bytes
            if res.restricted:
                restricted = True
                first_error = first_error or res.error
                break
            if not res.ok:
                first_error = first_error or res.error
                continue
            rows.extend(res.rows)

        if restricted:
            return ManifestEntry(key, ep.name, ticker, period, STATUS_RESTRICTED,
                                 calls=calls, n_bytes=n_bytes, error=first_error)
        if not rows:
            status = STATUS_ERROR if first_error else STATUS_EMPTY
            return ManifestEntry(key, ep.name, ticker, period, status,
                                 calls=calls, n_bytes=n_bytes, error=first_error)
        n_rows, first, last = self.archive.write(ep, rows, symbol=ticker, period=period)
        return ManifestEntry(key, ep.name, ticker, period, STATUS_OK, rows=n_rows,
                             calls=calls, n_bytes=n_bytes, first_date=first,
                             last_date=last, error=first_error)

    # ── symbol resolution ───────────────────────────────────────────────────

    def _resolve(self, ticker: str, price_ep: Endpoint) -> tuple[Resolution, ManifestEntry | None]:
        """Find a vendor symbol that returns bars, trying the literal ticker first."""
        candidates = candidate_symbols(ticker, renames=self.renames)
        attempts: list[str] = []
        last: ManifestEntry | None = None
        for candidate in candidates:
            attempts.append(candidate)
            entry = last = self._fetch_symbol_endpoint(price_ep, ticker, candidate, None)
            if entry.status == STATUS_OK and entry.rows:
                method = "direct" if candidate == ticker else "variant"
                if candidate != ticker and self.renames.get(ticker) == candidate:
                    method = "symbol_change"
                return (
                    Resolution(
                        ticker=ticker, resolved=candidate, method=method,
                        n_bars=entry.rows, first_date=entry.first_date,
                        last_date=entry.last_date, attempts=">".join(attempts),
                    ),
                    entry,
                )
            if entry.status == STATUS_RESTRICTED:
                return (
                    Resolution(ticker=ticker, method="restricted",
                               attempts=">".join(attempts),
                               note=(entry.error or "")[:160]),
                    entry,
                )
        # Record the miss so a resumed run does not re-try every candidate for a
        # name FMP simply does not carry (there are ~100 of those in a 2004+ pull).
        # `--no-retry-errors` is what pins a transport failure the same way.
        if last is not None:
            last.error = last.error or f"no bars for any of {attempts}"
        return (
            Resolution(ticker=ticker, method="unresolved", attempts=">".join(attempts)),
            last,
        )

    def _coverage(self, resolution: Resolution, ticker: str) -> None:
        """Attach the membership-window coverage figure to a resolution."""
        if self.spells is None:
            return
        windows = spell_windows(self.spells, ticker, self._start_ts, self._end_ts)
        if windows:
            resolution.spell_start = str(windows[0][0].date())
            resolution.spell_end = str(windows[-1][1].date())
        bars = None
        if resolution.resolved:
            frame = self.archive.read("prices_adjusted", ticker)
            if frame is not None and isinstance(frame.index, pd.DatetimeIndex):
                bars = frame.index
        resolution.spell_coverage = spell_coverage(bars, windows)

    # ── per-symbol driver ───────────────────────────────────────────────────

    def _process_symbol(
        self, ticker: str, endpoints: list[Endpoint]
    ) -> tuple[Resolution, list[ManifestEntry], int]:
        entries: list[ManifestEntry] = []
        skipped = 0
        price_ep = next((e for e in endpoints if e.name == "prices_adjusted"), None)
        price_done = False

        resolution = Resolution(ticker=ticker, resolved=ticker, method="assumed")
        if price_ep is not None:
            if self.archive.is_done(price_ep.key(ticker),
                                    retry_errors=self.config.retry_errors):
                prior = self.archive.read("prices_adjusted", ticker)
                has_bars = prior is not None and len(prior) > 0
                dated = has_bars and isinstance(prior.index, pd.DatetimeIndex)
                resolution = Resolution(
                    ticker=ticker,
                    resolved=ticker if has_bars else None,
                    method="cached" if has_bars else "unresolved",
                    n_bars=len(prior) if prior is not None else 0,
                    # Carry the vendor's date range through on a resumed run too —
                    # comparing it to the membership window is what exposes a
                    # ticker later reused by a different company (PLL, PD, ONE).
                    first_date=str(prior.index.min().date()) if dated else None,
                    last_date=str(prior.index.max().date()) if dated else None,
                )
                price_done, skipped = True, skipped + 1
            else:
                resolution, entry = self._resolve(ticker, price_ep)
                if entry is not None:
                    entries.append(entry)
                    price_done = True

        vendor_symbol = resolution.resolved
        if vendor_symbol is None:
            # Nothing FMP will serve for this name; record the miss and move on.
            self._coverage(resolution, ticker)
            return resolution, entries, skipped

        for ep in endpoints:
            for period in work_units(ep):
                if ep is price_ep and period is None and price_done:
                    continue  # already handled during resolution
                if self.archive.is_done(ep.key(ticker, period),
                                        retry_errors=self.config.retry_errors):
                    skipped += 1
                    continue
                entries.append(
                    self._fetch_symbol_endpoint(ep, ticker, vendor_symbol, period))

        self._coverage(resolution, ticker)
        return resolution, entries, skipped

    # ── run ─────────────────────────────────────────────────────────────────

    def run(self, symbols: list[str] | None = None) -> DownloadReport:
        symbols = symbols if symbols is not None else self.universe()
        endpoints = self.endpoints()
        self.archive.load_manifest()

        self.report.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.report.symbols = len(symbols)
        self.report.units_total = sum(len(work_units(e)) for e in endpoints) * len(symbols)
        log.info("downloading %d symbols x %d endpoints (%d units) at %d calls/min",
                 len(symbols), len(endpoints), self.report.units_total,
                 self.client.limiter.per_minute)

        t0 = time.monotonic()
        done_symbols = 0
        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool:
            futures = {
                pool.submit(self._process_symbol, sym, endpoints): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    resolution, entries, skipped = future.result()
                except Exception as e:  # noqa: BLE001 — one symbol must not kill the run
                    log.warning("symbol %s failed: %s", sym, e)
                    self.report.resolutions.append(
                        Resolution(ticker=sym, method="crashed", note=str(e)[:200]))
                    continue
                for entry in entries:
                    self.archive.record(entry)
                    self.report.bump(entry.status)
                    self.report.units_done += 1
                self.report.units_skipped += skipped
                self.report.resolutions.append(resolution)
                done_symbols += 1
                if done_symbols % 25 == 0 or done_symbols == len(symbols):
                    stats = self.client.stats()
                    elapsed = time.monotonic() - t0
                    rate = done_symbols / elapsed * 60 if elapsed else 0
                    remaining = (len(symbols) - done_symbols) / rate if rate else 0
                    log.info(
                        "%d/%d symbols | %d calls | %.1f GB | %.0f sym/min | ~%.0f min left",
                        done_symbols, len(symbols), stats["calls"],
                        stats["bytes"] / 1e9, rate, remaining,
                    )

        self.report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.report.client_stats = self.client.stats()
        self.report.archive_mb = self.archive.disk_usage_mb()
        self.archive.compact_manifest({
            "config": {
                "start": self.config.start, "end": self.config.end,
                "groups": list(self.config.groups), "indices": list(self.config.indices),
                "symbols": len(symbols),
            },
            "client_stats": self.report.client_stats,
        })
        return self.report
