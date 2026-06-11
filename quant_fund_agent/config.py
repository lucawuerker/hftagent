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
import os
from dataclasses import dataclass, field
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
        """Build settings from defaults → ``quant.config.yaml`` → env overrides."""
        settings = cls()
        path = Path(config_path) if config_path else Path(CONFIG_FILENAME)
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
