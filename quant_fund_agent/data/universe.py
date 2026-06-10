"""Universe resolution — turn config into a concrete list of tickers.

A run's universe comes from one of two places in :class:`DataSettings`:
  * ``tickers`` — an explicit list (highest priority); or
  * ``universe_preset`` — the name of a bundled static list under
    ``data/universes/<name>.txt`` (e.g. ``"demo"``, ``"sp100"``).

``n_tickers`` then caps the result (front of the list).  File-based providers
like LOBSTER ignore this (their universe is the set of CSV directories on disk);
it drives the API providers (yfinance / FMP / …).

NOTE: bundled preset lists are point-in-time snapshots and are **not**
survivorship-corrected — see the header in each ``.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_fund_agent.config import DataSettings

PRESET_DIR = Path(__file__).parent / "universes"


def available_presets() -> list[str]:
    """Names of the bundled universe presets (sans ``.txt``)."""
    return sorted(p.stem for p in PRESET_DIR.glob("*.txt"))


def load_preset(name: str) -> list[str]:
    """Read a bundled preset's tickers (uppercased, comments/blanks stripped)."""
    path = PRESET_DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(
            f"Unknown universe preset {name!r}. Available: {available_presets()}."
        )
    return [
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_universe(data: "DataSettings") -> list[str]:
    """Concrete ticker list for an API provider, from explicit list or preset."""
    if data.tickers:
        symbols = [t.strip().upper() for t in data.tickers if t.strip()]
    elif data.universe_preset:
        symbols = load_preset(data.universe_preset)
    else:
        raise ValueError(
            "No universe configured: set data.tickers or data.universe_preset "
            f"(presets: {available_presets()})."
        )
    if data.n_tickers is not None and len(symbols) > data.n_tickers:
        symbols = symbols[: data.n_tickers]
    return symbols
