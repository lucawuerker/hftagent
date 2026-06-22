"""Portals that deliver raw LOBSTER day-files for a chunk.

Two implementations:

* :class:`PlaywrightLobsterPortal` — drives the LOBSTER web portal against the
  **real** site layout (confirmed by introspection): requests data via the
  *bulk* form (``requestBulkData.php``: a one-symbol-per-line list, a level, and
  an ``interval`` — **0 = raw message+orderbook**, >0 = sampled book every N
  seconds), then reads finished orders off ``mydata.php`` (each links to
  ``download.php?file=…&id=…``) and streams the ``.7z`` to disk. Login/2FA is
  done once by a human via :meth:`recon`; the cookie ``storageState`` is reused.
* :class:`WatchFolderPortal` — no browser: drop a downloaded archive/folder into
  a directory and it is picked up, extracted and converted. Reliable fallback.

The downloaded ``.7z`` is fed to :func:`converter.convert_archive`, which
streams it **one day at a time** so a 2-year raw archive never unzips in full.

Playwright is imported lazily::

    ./venv/bin/pip install playwright py7zr && ./venv/bin/python -m playwright install chromium
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import subprocess
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

log = logging.getLogger("lobster_ingest.portal")


# ── archive extraction (zip native; 7z via py7zr or the 7z CLI) ─────────────
def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract ``archive`` into ``dest`` and return ``dest``.

    Supports ``.zip`` natively. ``.7z`` (LOBSTER's format) needs ``py7zr`` or a
    ``7z``/``7za`` binary on PATH. For large archives prefer
    :func:`converter.convert_archive`, which streams day-by-day instead.
    """
    dest.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return dest
    if suffix == ".7z":
        try:
            import py7zr  # type: ignore

            with py7zr.SevenZipFile(archive, mode="r") as z:
                z.extractall(path=dest)
            return dest
        except ImportError:
            import shutil
            import subprocess

            exe = shutil.which("7z") or shutil.which("7za")
            if not exe:
                raise RuntimeError(
                    "extracting .7z needs `pip install py7zr` or a 7z binary on PATH"
                )
            subprocess.run([exe, "x", f"-o{dest}", "-y", str(archive)], check=True)
            return dest
    raise RuntimeError(f"unsupported archive type: {archive.name}")


def _flatten_to_csv_dir(root: Path) -> Path:
    """Return the directory that directly holds the day CSVs (raw or sampled)."""
    if any(root.glob("*_message_*.csv")) or any(root.glob("*.csv")):
        return root
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(sub.glob("*.csv")):
            return sub
    return root


# ── watch-folder portal (manual download, automatic everything else) ────────
@dataclass
class WatchFolderPortal:
    """Pick up a manually-downloaded chunk from ``drop_dir`` (no browser)."""

    drop_dir: str | Path
    poll_interval_s: int = 30
    poll_timeout_s: int = 24 * 3600

    def fetch(self, ticker: str, start: _dt.date, end: _dt.date, levels: int, dest: Path) -> Path:
        drop = Path(self.drop_dir)
        drop.mkdir(parents=True, exist_ok=True)
        stamp = f"{ticker}_{start.isoformat()}"
        log.info("waiting for a download matching '%s*' in %s …", stamp, drop)
        deadline = time.time() + self.poll_timeout_s
        while time.time() < deadline:
            for d in drop.glob(f"{ticker}_{start.isoformat()}*"):
                if d.is_dir() and any(d.glob("*.csv")):
                    return d
            for pat in (f"{ticker}_{start.isoformat()}*.zip", f"{ticker}_{start.isoformat()}*.7z"):
                for arc in drop.glob(pat):
                    log.info("extracting %s", arc.name)
                    extract_archive(arc, dest)
                    return _flatten_to_csv_dir(dest)
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"no chunk for {stamp} appeared in {drop} within timeout")


# ── Playwright portal ───────────────────────────────────────────────────────
@dataclass
class LobsterPortalConfig:
    """Live LOBSTER URLs + selectors (confirmed against the real site).

    Two request products:
    * **single** ``requestdata.php`` — the RAW message+orderbook reconstruction
      (no ``interval`` field) → all 19 ticker_data columns. This is what the GLD
      sample is and the product to use for the full microstructure panel.
    * **bulk** ``requestBulkData.php`` — a sampled / per-event *order book* only
      (has an ``interval`` field); it does NOT include the message/event stream,
      so the flow/trade columns cannot be derived. Useful only for book-only data.
    """

    login_url: str = "https://php.lobsterdata.com/SignIn.php"
    request_url: str = "https://php.lobsterdata.com/requestdata.php"   # single → RAW
    bulk_url: str = "https://php.lobsterdata.com/requestBulkData.php"  # bulk → sampled book only
    mydata_url: str = "https://php.lobsterdata.com/mydata.php"
    archive_url: str = "https://php.lobsterdata.com/data_archive.php"
    storage_state: str = "data/lobster_raw/.lobster_state.json"
    headless: bool = True

    # single request form (requestdata.php) → RAW message+orderbook
    sel_stock: str = "#stock1"
    sel_start: str = "#startdate1"
    sel_end: str = "#enddate1"
    sel_level: str = "#level"
    sel_submit: str = "#submit"

    # bulk request form (requestBulkData.php) → sampled order book only
    sel_bulk_start: str = "#startDate"
    sel_bulk_end: str = "#endDate"
    sel_bulk_level: str = "#level"
    sel_bulk_interval: str = "#interval"
    sel_bulk_project: str = "#projectTag"
    sel_bulk_symbols: str = "#symbolList"  # file upload, one symbol per line
    sel_bulk_submit: str = "button[type='submit']"

    # finished orders (mydata.php) — each ready order links to download.php
    sel_download_link: str = "a[href*='download.php']"


# Parse a LOBSTER download filename. RAW orders are ``TICKER_START_END_LEVEL.7z``;
# sampled orders append the snapshot interval: ``TICKER_START_END_LEVEL_INTERVAL.7z``.
_FNAME_RE = re.compile(
    r"([A-Za-z0-9.\-]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_(\d+)(?:_(\d+))?\.7z"
)
_ID_RE = re.compile(r"[?&]id=(\d+)")


@dataclass(frozen=True)
class Order:
    """A finished LOBSTER order ready to download off ``mydata.php``.

    ``interval`` is parsed from the filename: ``None`` (no suffix) means the RAW
    single-form product; a positive value is a sampled book. The converter
    auto-detects the actual contents, so this is only a hint.
    """

    ticker: str
    start: str
    end: str
    level: int
    interval: int | None      # None/0 → raw message+orderbook; >0 → sampled book
    url: str
    order_id: str

    @property
    def is_raw(self) -> bool:
        return not self.interval  # None or 0

    @property
    def key(self) -> str:
        itv = "raw" if self.is_raw else str(self.interval)
        return f"{self.ticker}_{self.start}_{self.end}_L{self.level}_{itv}_{self.order_id}"


class PlaywrightLobsterPortal:
    """Drive the LOBSTER web portal: request data, list + download finished orders."""

    def __init__(self, cfg: LobsterPortalConfig | None = None,
                 poll_interval_s: int = 60, poll_timeout_s: int = 6 * 3600):
        self.cfg = cfg or LobsterPortalConfig()
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s

    @staticmethod
    def _playwright():
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Playwright is not installed. Run:\n"
                "  ./venv/bin/pip install playwright py7zr\n"
                "  ./venv/bin/python -m playwright install chromium"
            ) from e
        return sync_playwright

    @contextmanager
    def session(self):
        """Yield a logged-in ``page`` (reuses the cookie storageState)."""
        sync_playwright = self._playwright()
        state = Path(self.cfg.storage_state)
        if not state.exists():
            raise RuntimeError(
                f"no saved session at {state}; run --recon first "
                "(see docs/lobster-ingestion/RUNBOOK.md)"
            )
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.cfg.headless)
            ctx = browser.new_context(storage_state=str(state), accept_downloads=True)
            page = ctx.new_page()
            try:
                yield page
            finally:
                try:
                    ctx.storage_state(path=str(state))  # refresh cookies
                except Exception:
                    pass
                browser.close()

    # -- one-time bootstrap: log in by hand, persist cookies --
    def recon(self) -> None:
        sync_playwright = self._playwright()
        state = Path(self.cfg.storage_state)
        state.parent.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context()
            page = ctx.new_page()
            page.goto(self.cfg.login_url)
            input(
                "\nLog in to LOBSTER in the opened browser (complete any 2FA/CAPTCHA),\n"
                "then press Enter here to save the session…\n"
            )
            ctx.storage_state(path=str(state))
            browser.close()
        log.info("saved LOBSTER session to %s", state)

    # -- request RAW data (single form; message+orderbook, all 19 columns) --
    def place_single_request(
        self, page, ticker: str, start: _dt.date, end: _dt.date, level: int,
    ) -> None:
        """Submit one ``requestdata.php`` request → RAW message+orderbook.

        This is the product that yields ALL ticker_data columns (the bulk form
        delivers only an order book, no message/event stream).
        """
        c = self.cfg
        page.goto(c.request_url, wait_until="load")
        page.fill(c.sel_stock, ticker)
        page.fill(c.sel_start, start.isoformat())
        page.fill(c.sel_end, end.isoformat())
        page.fill(c.sel_level, str(level))
        page.click(c.sel_submit)
        log.info("placed RAW request: %s %s→%s level %d", ticker, start, end, level)

    # -- request SAMPLED order book (bulk form; NO message/event data) --
    def place_bulk_request(
        self, page, tickers: list[str], start: _dt.date, end: _dt.date,
        level: int, interval: int = 0, project: str | None = None,
    ) -> None:
        """Submit a bulk request → **sampled order book only** (no flow columns).

        The bulk product contains only the order-book levels (snapshotted every
        ``interval`` seconds, or per-event at 0); it does NOT include the message
        stream, so ``trade/orderFlow/...`` cannot be derived. Use
        :meth:`place_single_request` for the full (raw) product.
        """
        c = self.cfg
        page.goto(c.bulk_url, wait_until="load")
        page.fill(c.sel_bulk_start, start.isoformat())
        page.fill(c.sel_bulk_end, end.isoformat())
        page.fill(c.sel_bulk_level, str(level))
        page.fill(c.sel_bulk_interval, str(interval))
        if project:
            page.fill(c.sel_bulk_project, project)
        # symbol list: one symbol per line, uploaded as a file
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(tickers) + "\n")
            symfile = fh.name
        page.set_input_files(c.sel_bulk_symbols, symfile)
        page.click(c.sel_bulk_submit)
        Path(symfile).unlink(missing_ok=True)
        log.info(
            "placed bulk request: %d tickers %s→%s level %d interval %d (%s)",
            len(tickers), start, end, level, interval, "RAW" if interval == 0 else "sampled",
        )

    # -- list finished orders on mydata.php --
    def list_ready_orders(self, page) -> list[Order]:
        c = self.cfg
        page.goto(c.mydata_url, wait_until="load")
        hrefs = page.eval_on_selector_all(
            c.sel_download_link, "els => els.map(e => e.href)"
        )
        orders: dict[str, Order] = {}
        for href in hrefs:
            fn = _FNAME_RE.search(href)
            oid = _ID_RE.search(href)
            if not fn:
                continue
            tkr, s, e, lvl, itv = fn.groups()
            order = Order(
                ticker=tkr, start=s, end=e, level=int(lvl),
                interval=(int(itv) if itv else None),  # no suffix → raw
                url=href, order_id=(oid.group(1) if oid else ""),
            )
            orders[order.key] = order
        return list(orders.values())

    # -- stream-download one order's .7z to disk --
    def download_order(self, page, order: Order, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / self._download_filename(order)
        if self._curl_download(page, order.url, target):
            log.info("downloaded %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
            return target

        # Navigating to an attachment URL fires a download and aborts the page
        # navigation — the try/except must sit *inside* expect_download so the
        # context manager still captures the download after goto raises.
        with page.expect_download(timeout=self.poll_timeout_s * 1000) as dl_info:
            try:
                page.goto(order.url)
            except Exception:
                pass
        dl = dl_info.value
        target = dest_dir / (dl.suggested_filename or target.name)
        dl.save_as(str(target))
        log.info("downloaded %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
        return target

    def delete_order(self, page, order: Order) -> None:
        """Remove a finished order from LOBSTER server storage."""
        c = self.cfg
        page.goto(c.archive_url, wait_until="load")
        checkbox = f"#delete{order.order_id}"
        if not page.locator(checkbox).count():
            log.info("archive order %s already absent", order.order_id)
            return
        page.check(checkbox)
        page.click("#submit")
        page.wait_for_load_state("load")
        log.info("deleted LOBSTER archive order %s", order.order_id)

    def _download_filename(self, order: Order) -> str:
        query = parse_qs(urlparse(order.url).query)
        file_param = query.get("file", [""])[0]
        name = Path(unquote(file_param)).name
        return name or f"{order.key}.7z"

    def _curl_download(self, page, url: str, target: Path) -> bool:
        cookies = page.context.cookies("https://php.lobsterdata.com")
        cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)
        if not cookie_header:
            return False
        for attempt in range(1, 6):
            target.unlink(missing_ok=True)
            cmd = [
                "curl",
                "--http1.1",
                "-L",
                "--fail",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "100000",
                "--speed-time",
                "120",
                "--cookie",
                cookie_header,
                "-o",
                str(target),
                url,
            ]
            log.info("curl download attempt %d/5 for %s", attempt, target.name)
            try:
                subprocess.run(cmd, check=True)
                return target.exists() and target.stat().st_size > 0
            except (OSError, subprocess.CalledProcessError) as exc:
                log.warning("curl download attempt %d failed for %s: %s",
                            attempt, target.name, exc)
                target.unlink(missing_ok=True)
                time.sleep(10)
        return False
