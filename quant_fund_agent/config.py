"""Central configuration for QuantFundAgent.

Historically every module read its own ``os.getenv(...)`` at import time and the
data directory was hardcoded to LOBSTER's ``ticker_data/``.  To make the project
runnable against *any* market-data vendor, configuration is consolidated here in
one serialisable :class:`Settings` object.

Precedence (highest first):

1. **Explicit environment variables** — preserves every existing script/test
   and the simulation's ``os.environ["DATA_DIR"] = ...`` handshake.
2. **``quant.config.yaml``** — written by the onboarding wizard
   (``python -m quant_fund_agent.setup``); see ``docs/data-layer/ONBOARDING.md``.
3. **Built-in defaults** — the LOBSTER ``ticker_data/`` setup, so a fresh clone
   with no config file behaves exactly as before.

This module is intentionally dependency-light: PyYAML is imported lazily and only
when a config file actually exists, so nothing new is required to run the legacy
LOBSTER path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default location of the (optional) user config file.
CONFIG_FILENAME = "quant.config.yaml"


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        n = int(raw)
        return n if n > 0 else None
    except ValueError:
        return None


@dataclass
class DataSettings:
    """Everything about *where the market data comes from*."""

    provider: str = "lobster"          # lobster | yfinance | fmp | alphavantage
    asset_class: str = "equity"        # equity (crypto/FX in a later phase)
    frequency: str = "1d"             # 1d | 1min | 10s | …
    data_dir: str = "ticker_data"      # LOBSTER CSV root (provider="lobster")
    cache_dir: str = "data/market"     # parquet cache root (API providers)
    n_tickers: int | None = None       # optional universe cap (memory)
    tickers: list[str] | None = None   # explicit ticker list (None = auto)
    universe_preset: str | None = None # e.g. "sp100" (resolved in data/universe.py)
    start: str | None = None           # ISO date, inclusive (API providers)
    end: str | None = None             # ISO date, exclusive (API providers)
    dtype: str = "float32"             # panel numeric precision

    # Non-OHLCV (fundamentals / estimates / events) enrichment — equity only.
    fundamentals: bool = True          # off-switch (also QF_FUNDAMENTALS=0)
    reporting_lag_days: int = 60       # availability = fiscal-end + lag (no filing date)
    fundamentals_staleness_days: int = 400  # drop a fundamental older than this → NaN
    fundamentals_ttl_days: int = 90    # quarterly cache refresh for slow-moving data


@dataclass
class Settings:
    """Top-level configuration object passed around the data layer."""

    data: DataSettings = field(default_factory=DataSettings)

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "Settings":
        """Build settings from defaults → config file → env overrides.

        The config file is ``config_path`` if given, else ``QF_CONFIG_FILE`` (so
        ``run_fund --config other.yaml`` can run the whole data layer under an
        alternate config), else ``quant.config.yaml``.
        """
        settings = cls()
        chosen = config_path or os.getenv("QF_CONFIG_FILE") or CONFIG_FILENAME
        path = Path(chosen)
        if path.exists():
            settings._apply_yaml(path)
        settings._apply_env()
        return settings

    def _apply_yaml(self, path: Path) -> None:
        try:
            import yaml  # lazy: only needed when a config file exists
        except ImportError as e:  # pragma: no cover - helpful error
            raise RuntimeError(
                f"Found {path} but PyYAML is not installed. "
                "Run `./venv/bin/pip install pyyaml`."
            ) from e
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        data = raw.get("data", {}) or {}
        for key, value in data.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)

    def _apply_env(self) -> None:
        """Environment variables override the file (back-compat with legacy code)."""
        if os.getenv("QF_DATA_PROVIDER"):
            self.data.provider = os.environ["QF_DATA_PROVIDER"]
        if os.getenv("DATA_DIR"):
            self.data.data_dir = os.environ["DATA_DIR"]
        n = _env_int("ARCHITECT_N_TICKERS")
        if n is not None:
            self.data.n_tickers = n

    # ── helpers ─────────────────────────────────────────────────────────

    def with_data_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy with non-``None`` ``DataSettings`` fields overridden.

        Used by :func:`quant_fund_agent.data.load_panel` so a call site that
        passes an explicit ``data_dir`` / ``n_tickers`` / ``fields`` still wins
        over the ambient config, without mutating the shared object.
        """
        applied = {k: v for k, v in overrides.items() if v is not None}
        if not applied:
            return self
        return dataclasses.replace(self, data=dataclasses.replace(self.data, **applied))


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Convenience entry point: load settings fresh (cheap; reads env each call).

    Not cached, so the simulation's ``os.environ["DATA_DIR"] = ...`` handshake and
    per-test env tweaks are always reflected.  The heavy panel load it feeds is
    cached by the callers, so this runs rarely.
    """
    return Settings.load(config_path)


# ── config identity & provenance ────────────────────────────────────────────
#
# A factor or strategy is meaningful only relative to the data it was researched
# on (provider / asset-class / universe / timespan).  These helpers give that
# data config a stable identity so a (config, prerun) *scope* can be named on
# disk and stamped onto every record for reproducibility.  They span exactly the
# ``DataSettings`` dimensions that change panel *content* — the same set the
# research panel cache keys on (see ``research_service._panel_cache_key``).

_FINGERPRINT_KEYS = (
    "provider", "asset_class", "frequency", "universe_preset",
    "tickers", "start", "end", "n_tickers", "dtype",
)


def config_fingerprint(data: DataSettings) -> str:
    """Stable 12-char hash of the panel-defining config dimensions.

    Two configs with the same fingerprint produce the same panel; a change to
    provider / asset-class / frequency / universe / timespan / dtype changes it.
    """
    blob = json.dumps(
        {k: getattr(data, k) for k in _FINGERPRINT_KEYS},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _slug(text: str) -> str:
    """Filesystem-safe lowercase slug (``yfinance/equity`` → ``yfinance_equity``)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return s or "x"


def default_config_name(data: DataSettings) -> str:
    """Human-readable default scope name derived from the config.

    ``<provider>_<asset_class>_<universe>`` (e.g. ``yfinance_equity_demo``).  Two
    configs that differ only in timespan share a name; the per-scope config
    snapshot + hash mismatch warning surface that, and ``--config-name`` overrides.
    """
    universe = data.universe_preset or (
        f"custom{len(data.tickers)}" if data.tickers else "custom"
    )
    return "_".join(_slug(p) for p in (data.provider, data.asset_class, universe))


def provenance_stamp(scope_label: str, data: DataSettings) -> dict[str, Any]:
    """Record of *which scope + data config* a factor/strategy was born under."""
    return {
        "scope": scope_label,
        "config_hash": config_fingerprint(data),
        "provider": data.provider,
        "asset_class": data.asset_class,
        "frequency": data.frequency,
        "universe": data.universe_preset,
        "timespan": [data.start, data.end],
        "stamped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
