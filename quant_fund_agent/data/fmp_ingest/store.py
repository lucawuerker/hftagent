"""The local FMP archive: parquet layout, atomic writes and resume state.

Layout under ``<root>`` (default ``data/vendor/fmp``)::

    prices/adjusted/<SYM>.parquet          prices/unadjusted/<SYM>.parquet
    prices/dividends/<SYM>.parquet         prices/splits/<SYM>.parquet
    fundamentals/income-statement/quarter/<SYM>.parquet
    fundamentals/ratios/annual/<SYM>.parquet            …
    reference/profile/<SYM>.parquet        reference/market-cap/<SYM>.parquet
    index/historical-sp500-constituent/_global.parquet  …
    manifest.jsonl   manifest.json   capabilities.json   symbol_map.csv

Every column the vendor returns is kept (Layer A): the archive is the source of
truth, so adding a canonical panel field later never means re-downloading.

**Resume** is the property that matters for a multi-hour, ~5 GB pull.  One unit
of work is a ``(endpoint, period, symbol)`` triple — possibly several HTTP calls
when a date range is chunked — and it lands as exactly one parquet file plus one
manifest row.  The manifest is an append-only JSONL journal (crash-safe under
concurrent workers, no rewrite-the-world on every update) that is compacted into
``manifest.json`` at the end of a run; loading reads the compacted file and
overlays the journal.  A killed run therefore re-enters exactly where it stopped.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_fund_agent.data.fmp_ingest.endpoints import Endpoint

log = logging.getLogger("data.fmp_ingest.store")

DEFAULT_ROOT = Path("data/vendor/fmp")

#: filename used for a ``kind="global"`` endpoint (no symbol dimension)
GLOBAL_STEM = "_global"

#: manifest statuses.  ``restricted`` is terminal for the current plan;
#: ``error`` is retried on the next run, ``ok``/``empty`` are not.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_RESTRICTED = "restricted"
STATUS_ERROR = "error"
_TERMINAL = (STATUS_OK, STATUS_EMPTY, STATUS_RESTRICTED)


@dataclass
class ManifestEntry:
    """One unit of work's outcome — the resume record."""

    key: str
    endpoint: str
    symbol: str | None = None
    period: str | None = None
    status: str = STATUS_OK
    rows: int = 0
    calls: int = 0
    n_bytes: int = 0
    first_date: str | None = None
    last_date: str | None = None
    error: str | None = None
    fetched_at: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    @property
    def done(self) -> bool:
        return self.status in _TERMINAL


def _safe(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def coerce_frame(rows: list[dict]) -> pd.DataFrame:
    """Vendor JSON rows → a parquet-safe frame, preserving every column.

    Object columns are pushed to numeric where the whole column parses, and to
    pandas ``string`` otherwise, so mixed ``None``/number/text columns (FMP has
    plenty) never blow up the parquet writer or silently become ``object``.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype != object:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna()
        # Convert only when nothing that *was* present failed to parse — that
        # keeps genuine labels ("USD", "FY", "Q3") as text.
        if non_null.any() and (numeric.notna() | ~non_null).all():
            df[col] = numeric
        else:
            df[col] = df[col].astype("string")
    return df


def _dedup(df: pd.DataFrame, how: str) -> pd.DataFrame:
    """Collapse repeats according to the endpoint's ``dedup`` policy.

    ``"date"`` treats the stamp as the primary key (a price bar, a fiscal
    period); ``"row"`` keeps every distinct row on a date, which index change
    logs need — several names can enter and leave on one effective date, and
    keeping only the last silently drops the rest of that day's events.
    """
    if df.empty:
        return df
    if how == "row":
        return df[~df.reset_index().duplicated().to_numpy()]
    return df[~df.index.duplicated(keep="last")]


def _index_frame(df: pd.DataFrame, date_field: str | None, dedup: str = "date") -> pd.DataFrame:
    """Set/sort the date index and collapse repeats per the dedup policy."""
    if df.empty or not date_field or date_field not in df.columns:
        return df
    out = df.copy()
    out[date_field] = pd.to_datetime(out[date_field], errors="coerce")
    out = out[out[date_field].notna()]
    if out.empty:
        return out
    out = out.set_index(date_field).sort_index()
    return _dedup(out, dedup)


class Archive:
    """Filesystem archive + manifest.  Safe to share across worker threads."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        self._entries: dict[str, ManifestEntry] = {}
        self._journal = self.root / "manifest.jsonl"
        self._compact = self.root / "manifest.json"

    # ── paths ───────────────────────────────────────────────────────────────

    def path_for(
        self, endpoint: Endpoint, symbol: str | None = None, period: str | None = None
    ) -> Path:
        stem = _safe(symbol) if symbol else GLOBAL_STEM
        return self.root / endpoint.dest_for(period) / f"{stem}.parquet"

    # ── manifest ────────────────────────────────────────────────────────────

    def load_manifest(self) -> dict[str, ManifestEntry]:
        """Read compacted manifest, then overlay the journal (journal wins)."""
        entries: dict[str, ManifestEntry] = {}
        if self._compact.exists():
            try:
                raw = json.loads(self._compact.read_text())
                for key, payload in (raw.get("entries") or {}).items():
                    entries[key] = ManifestEntry(**payload)
            except Exception as e:  # noqa: BLE001 — a corrupt manifest must not
                log.warning("manifest.json unreadable (%s); relying on journal", e)
        if self._journal.exists():
            for line in self._journal.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    entries[payload["key"]] = ManifestEntry(**payload)
                except Exception:  # noqa: BLE001 — skip a torn final line
                    continue
        with self._lock:
            self._entries = entries
        return dict(entries)

    def record(self, entry: ManifestEntry) -> None:
        """Append one outcome to the journal (thread-safe, crash-safe)."""
        with self._lock:
            self._entries[entry.key] = entry
            self._journal.parent.mkdir(parents=True, exist_ok=True)
            with self._journal.open("a") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")

    def compact_manifest(self, extra: dict | None = None) -> Path:
        """Fold the journal into ``manifest.json`` and truncate it."""
        with self._lock:
            entries = dict(self._entries)
        payload = {
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_entries": len(entries),
            **(extra or {}),
            "entries": {k: asdict(v) for k, v in entries.items()},
        }
        self._compact.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._compact.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self._compact)
        if self._journal.exists():
            self._journal.unlink()
        return self._compact

    def is_done(self, key: str, *, retry_errors: bool = True) -> bool:
        """Whether a unit of work can be skipped on a resumed run."""
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.status == STATUS_ERROR:
            return not retry_errors
        return entry.done

    # ── data ────────────────────────────────────────────────────────────────

    def write(
        self,
        endpoint: Endpoint,
        rows: list[dict],
        *,
        symbol: str | None = None,
        period: str | None = None,
        merge: bool = True,
    ) -> tuple[int, str | None, str | None]:
        """Write (or merge into) one archive file.

        ``merge`` unions with what is already on disk — required for windowed
        endpoints, whose chunks arrive as separate requests.  Returns
        ``(n_rows, first_date, last_date)``.
        """
        df = _index_frame(coerce_frame(rows), endpoint.date_field, endpoint.dedup)
        path = self.path_for(endpoint, symbol, period)
        if merge and path.exists() and endpoint.date_field:
            existing = self.read_path(path)
            if existing is not None and not existing.empty:
                df = _dedup(pd.concat([existing, df]).sort_index(), endpoint.dedup)
        if df.empty:
            return 0, None, None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(path)  # atomic on the same filesystem
        first = last = None
        if isinstance(df.index, pd.DatetimeIndex) and len(df):
            first = str(df.index.min().date())
            last = str(df.index.max().date())
        return len(df), first, last

    @staticmethod
    def read_path(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001 — a torn file is a cache miss
            log.warning("archive read failed (%s): %s", path, e)
            return None

    def read(
        self, endpoint: Endpoint | str, symbol: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame | None:
        """Read one archive file (``endpoint`` may be a registry name)."""
        if isinstance(endpoint, str):
            from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS

            endpoint = ENDPOINTS[endpoint]
        return self.read_path(self.path_for(endpoint, symbol, period))

    def symbols_present(self, endpoint: Endpoint | str, period: str | None = None) -> list[str]:
        """Tickers with a file for ``endpoint`` (drives offline panel loads)."""
        if isinstance(endpoint, str):
            from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS

            endpoint = ENDPOINTS[endpoint]
        directory = self.root / endpoint.dest_for(period)
        if not directory.exists():
            return []
        return sorted(p.stem for p in directory.glob("*.parquet") if p.stem != GLOBAL_STEM)

    # ── misc ────────────────────────────────────────────────────────────────

    def write_json(self, name: str, payload: dict | list) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(path)
        return path

    def read_json(self, name: str) -> dict | list | None:
        path = self.root / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("could not read %s: %s", path, e)
            return None

    def disk_usage_mb(self) -> float:
        if not self.root.exists():
            return 0.0
        total = sum(p.stat().st_size for p in self.root.rglob("*.parquet"))
        return round(total / 1e6, 1)
