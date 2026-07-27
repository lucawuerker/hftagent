"""Offline provider over the local FMP premium archive.

This is the consumption half of ``data/fmp_ingest/`` (the acquisition half).  It
reads the parquet archive on disk and emits the ordinary panel contract
``{field: DataFrame(index × tickers)}`` — **no network, ever**, so a research run,
a walk-forward backtest and a re-run six months from now all see byte-identical
data.  Select it with ``data.provider: fmp_archive``.

Two things make it more than a cache reader:

**1. Every field the archive holds, not the 13 the live API path normalises.**
The canonical vocabulary comes from :data:`data.fields.ARCHIVE_FIELD_SPECS`
(~130 fundamental fields), and ``available_fields`` reports only what is actually
on disk — so factor gating stays honest about a partial download.

**2. A stricter point-in-time stamp.**  FMP's ``ratios`` / ``key-metrics`` /
``financial-growth`` endpoints carry no filing date, only the fiscal period end.
Stamping those at ``period_end + 60 days`` (the generic fallback) is both
imprecise and occasionally *leaking* — plenty of filings land later than that.
Here they inherit the **actual ``filingDate``/``acceptedDate`` of the matching
income statement**, joined on ``(fiscalYear, period)``, and fall back to the lag
only when no statement row matches.  Everything then flows through the existing
:mod:`data.fundamentals` availability→forward-fill machinery, so
``_truncate_as_of`` keeps working unchanged.

Memory note: the full universe is ~1 300 tickers × ~5 600 daily bars, so one
field is ~30 MB in float32 and *all* fields would be several GB.  ``load`` filters
to the requested ``fields`` **before** materialising anything, so targeted loads
(``load_panel(fields=[...])``) only pay for what they ask for.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import pandas as pd

from quant_fund_agent.data.fields import (
    ARCHIVE_FIELD_SPECS,
    CATEGORICAL_FIELDS,
    NON_OHLCV_FIELDS,
    archive_map_for,
    coerce_numeric,
    normalize,
    pick,
)
from quant_fund_agent.data.fundamentals import (
    DEFAULT_REPORTING_LAG_DAYS,
    DEFAULT_STALENESS_CAP_DAYS,
    align_to_index,
    availability_date,
    build_record_frame,
    parse_date,
)
from quant_fund_agent.data.providers.base import DataProvider
from quant_fund_agent.data.tiers import TIERS

log = logging.getLogger("data.providers.fmp_archive")

#: adjusted-OHLCV column names in the archive → canonical panel fields
_PRICE_COLMAP = {
    "adjOpen": "open", "adjHigh": "high", "adjLow": "low",
    "adjClose": "close", "volume": "volume",
}
_OHLCV = ("open", "high", "low", "close", "volume")

#: statement endpoints that carry a real filing date (the availability anchors)
_FILED_SOURCES = ("income_statement", "balance_sheet", "cash_flow")
#: endpoints with no filing date — they inherit one via (fiscalYear, period)
_UNFILED_SOURCES = ("ratios", "key_metrics", "financial_growth")

_FILING_KEYS = ("filingDate", "fillingDate", "acceptedDate")
_PERIOD_END_KEYS = ("date", "fiscalDateEnding")

#: sector/industry are (near-)constant labels: knowable throughout any backtest
_STATIC_AVAILABILITY = pd.Timestamp("1990-01-01")


class FMPArchiveProvider(DataProvider):
    """Panel provider backed by ``data/vendor/fmp`` (see ``fmp_ingest``)."""

    name = "fmp_archive"
    asset_classes = ("equity",)

    def __init__(self, settings) -> None:  # noqa: ANN001 — mirrors DataProvider
        super().__init__(settings)
        from quant_fund_agent.data.fmp_ingest.store import DEFAULT_ROOT, Archive

        root = os.getenv("QF_FMP_ARCHIVE") or getattr(
            self.data, "archive_dir", None) or DEFAULT_ROOT
        self.archive = Archive(root)
        self.period = str(getattr(self.data, "fundamentals_period", "quarter"))

    # ── capability ──────────────────────────────────────────────────────────

    def available_fields(self) -> frozenset[str]:
        """OHLCV plus every canonical field whose source endpoint is on disk."""
        fields = set(TIERS["standard"])
        present = self._present_sources()
        for spec in ARCHIVE_FIELD_SPECS:
            if spec.source in present:
                fields.add(spec.name)
        if not getattr(self.data, "fundamentals", True) or os.getenv("QF_FUNDAMENTALS") == "0":
            fields -= NON_OHLCV_FIELDS
        return frozenset(fields)

    def _present_sources(self) -> frozenset[str]:
        """Endpoint names with at least one parquet file in the archive."""
        from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS

        out = set()
        for spec in ARCHIVE_FIELD_SPECS:
            if spec.source in out or spec.source not in ENDPOINTS:
                continue
            ep = ENDPOINTS[spec.source]
            for period in (self.period, None, "annual"):
                directory = self.archive.root / ep.dest_for(period if ep.periods else None)
                if directory.exists() and any(directory.glob("*.parquet")):
                    out.add(spec.source)
                    break
        return frozenset(out)

    # ── data ────────────────────────────────────────────────────────────────

    def load(self, *, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
        from quant_fund_agent.data.universe import resolve_universe

        if not (self.data.start and self.data.end):
            raise ValueError(
                "fmp_archive needs data.start and data.end "
                "(run `python -m quant_fund_agent.setup` to write quant.config.yaml)."
            )
        symbols = [s for s in resolve_universe(self.data) if self._has_prices(s)]
        if not symbols:
            raise FileNotFoundError(
                f"No archived price files under {self.archive.root} for the configured "
                "universe. Run `scripts/fmp_bulk_download.py` first "
                "(see docs/data-layer/FMP_PREMIUM_ARCHIVE.md)."
            )

        wanted = set(fields) if fields is not None else set(self.available_fields())
        price_fields = [f for f in _OHLCV if f in wanted]
        # OHLC is needed as a scaffold even when only fundamentals are requested:
        # the daily index every record aligns onto comes from the price panel.
        panel = self._load_prices(symbols, price_fields or ["close"])
        if not panel:
            return {}
        index = next(iter(panel.values())).index

        fundamental_wanted = sorted(wanted & NON_OHLCV_FIELDS)
        if fundamental_wanted and self._fundamentals_enabled():
            panel.update(self._load_fundamentals(symbols, index, fundamental_wanted))

        if fields is not None:
            panel = {k: v for k, v in panel.items() if k in wanted}
        return panel

    def _fundamentals_enabled(self) -> bool:
        if not getattr(self.data, "fundamentals", True):
            return False
        return os.getenv("QF_FUNDAMENTALS", "1") != "0"

    def _has_prices(self, symbol: str) -> bool:
        return self.archive.path_for(
            _endpoint("prices_adjusted"), symbol).exists()

    # ── prices ──────────────────────────────────────────────────────────────

    def _load_prices(
        self, symbols: list[str], price_fields: list[str]
    ) -> dict[str, pd.DataFrame]:
        start, end = pd.Timestamp(self.data.start), pd.Timestamp(self.data.end)
        ep = _endpoint("prices_adjusted")
        cols: dict[str, dict[str, pd.Series]] = {}
        for sym in symbols:
            df = self.archive.read(ep, sym)
            if df is None or df.empty:
                continue
            df = df.rename(columns=_PRICE_COLMAP)
            window = df.loc[(df.index >= start) & (df.index <= end)]
            if window.empty:
                continue
            for field in price_fields:
                if field in window.columns:
                    series = pd.to_numeric(window[field], errors="coerce")
                    cols.setdefault(field, {})[sym] = series[~series.index.duplicated()]
        dtype = getattr(self.data, "dtype", None) or "float64"
        out: dict[str, pd.DataFrame] = {}
        for field, series_by_sym in cols.items():
            frame = pd.DataFrame(series_by_sym).sort_index()
            out[field] = frame.astype(dtype)
        return out

    # ── fundamentals ────────────────────────────────────────────────────────

    def _load_fundamentals(
        self, symbols: list[str], index: pd.DatetimeIndex, wanted: list[str]
    ) -> dict[str, pd.DataFrame]:
        needed_sources = sorted({
            spec.source for spec in ARCHIVE_FIELD_SPECS if spec.name in set(wanted)
        })
        lag = int(getattr(self.data, "reporting_lag_days", DEFAULT_REPORTING_LAG_DAYS))

        records: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            frame = self._records_for_symbol(sym, needed_sources, lag)
            if frame is not None and not frame.empty:
                records[sym] = frame
        if not records:
            return {}

        aligned = align_to_index(
            records, index,
            staleness_cap_days=int(getattr(self.data, "fundamentals_staleness_days",
                                           DEFAULT_STALENESS_CAP_DAYS)),
        )
        return {k: v for k, v in aligned.items() if k in set(wanted)}

    def _records_for_symbol(
        self, symbol: str, sources: list[str], lag: int
    ) -> pd.DataFrame | None:
        """One symbol's availability-stamped record frame across all sources."""
        rows: list[dict] = []
        filings = self._filing_dates(symbol) if any(
            s in _UNFILED_SOURCES for s in sources) else {}

        for source in sources:
            if source == "profile":
                rows += self._profile_rows(symbol)
            elif source == "market_cap":
                rows += self._market_cap_rows(symbol)
            elif source == "earnings":
                rows += self._earnings_rows(symbol)
            else:
                rows += self._statement_rows(symbol, source, filings, lag)
        return build_record_frame(rows)

    def _read_periodic(self, source: str, symbol: str) -> pd.DataFrame | None:
        """Preferred fiscal period for a statement-shaped endpoint, else annual."""
        ep = _endpoint(source)
        for period in (self.period, "annual") if ep.periods else (None,):
            frame = self.archive.read(ep, symbol, period)
            if frame is not None and not frame.empty:
                return frame
        return None

    def _filing_dates(self, symbol: str) -> dict[tuple, pd.Timestamp]:
        """``(fiscalYear, period) → filing date`` from the filed statements.

        This is what lets the unfiled endpoints inherit a *real* availability
        date instead of a fixed reporting-lag guess.
        """
        out: dict[tuple, pd.Timestamp] = {}
        for source in _FILED_SOURCES:
            frame = self._read_periodic(source, symbol)
            if frame is None:
                continue
            for record in frame.reset_index().to_dict("records"):
                filed = parse_date(pick(record, _FILING_KEYS))
                if filed is None:
                    continue
                key = (str(record.get("fiscalYear")), str(record.get("period")))
                # Earliest filing wins: a later restatement must not push the
                # availability date forward and hide a value that WAS public.
                if key not in out or filed < out[key]:
                    out[key] = filed
        return out

    def _statement_rows(
        self, symbol: str, source: str, filings: dict[tuple, pd.Timestamp], lag: int
    ) -> list[dict]:
        frame = self._read_periodic(source, symbol)
        if frame is None:
            return []
        field_map = archive_map_for(source)
        if not field_map:
            return []
        out: list[dict] = []
        for record in frame.reset_index().to_dict("records"):
            avail = parse_date(pick(record, _FILING_KEYS))
            if avail is None and source in _UNFILED_SOURCES:
                avail = filings.get((str(record.get("fiscalYear")), str(record.get("period"))))
            if avail is None:
                avail = availability_date(None, pick(record, _PERIOD_END_KEYS),
                                          reporting_lag_days=lag)
            values = normalize(record, field_map)
            if avail is not None and values:
                out.append({"availability": avail, **values})
        return out

    def _profile_rows(self, symbol: str) -> list[dict]:
        frame = self.archive.read(_endpoint("profile"), symbol)
        if frame is None or frame.empty:
            return []
        values = normalize(frame.reset_index().to_dict("records")[0],
                           archive_map_for("profile"))
        return [{"availability": _STATIC_AVAILABILITY, **values}] if values else []

    def _market_cap_rows(self, symbol: str) -> list[dict]:
        """Daily market cap — each observation is knowable on its own date."""
        frame = self.archive.read(_endpoint("market_cap"), symbol)
        if frame is None or frame.empty or "marketCap" not in frame.columns:
            return []
        series = pd.to_numeric(frame["marketCap"], errors="coerce").dropna()
        return [{"availability": stamp, "marketCap": float(value)}
                for stamp, value in series.items()]

    def _earnings_rows(self, symbol: str) -> list[dict]:
        """Estimates, actuals and surprises, all stamped at the report date.

        The surprise is only knowable once the actual prints, so the report date
        is the correct (conservative, non-leaking) availability for the estimate
        as well — the same rule the live FMP provider uses.
        """
        frame = self.archive.read(_endpoint("earnings"), symbol)
        if frame is None or frame.empty:
            return []
        out: list[dict] = []
        for record in frame.reset_index().to_dict("records"):
            avail = parse_date(pick(record, ("date",)))
            if avail is None:
                continue
            values = normalize(record, archive_map_for("earnings"))
            eps_actual = coerce_numeric(pick(record, ("epsActual", "eps")))
            eps_est = coerce_numeric(pick(record, ("epsEstimated", "epsEstimate")))
            rev_actual = coerce_numeric(pick(record, ("revenueActual", "revenue")))
            rev_est = coerce_numeric(pick(record, ("revenueEstimated", "estimatedRevenue")))
            if eps_actual is not None and eps_est is not None:
                values["epsSurprise"] = eps_actual - eps_est
            if rev_actual is not None and rev_est is not None:
                values["revenueSurprise"] = rev_actual - rev_est
            if values:
                out.append({"availability": avail, **values})
        return out


@lru_cache(maxsize=None)
def _endpoint(name: str):
    from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS

    return ENDPOINTS[name]


__all__ = ["FMPArchiveProvider", "CATEGORICAL_FIELDS"]
