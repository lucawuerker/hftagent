"""Universe resolution — turn config into a concrete list of tickers.

A run's universe comes from one of two places in :class:`DataSettings`:
  * ``tickers`` — an explicit list (highest priority); or
  * ``universe_preset`` — the name of a bundled static list under
    ``data/universes/<name>.txt`` (e.g. ``"demo"``, ``"sp100"``).

``n_tickers`` then caps the result (front of the list).  File-based providers
like LOBSTER ignore this (their universe is the set of CSV directories on disk);
it drives the API providers (yfinance / FMP / …).

NOTE: bundled preset ``.txt`` lists are *static* snapshots and are **not**
survivorship-corrected.  For a survivorship-bias-free, point-in-time universe set
``DataSettings.membership`` (e.g. ``"sp500"``) instead — see
:mod:`quant_fund_agent.data.membership` and ``docs/data-layer/SP500_MEMBERSHIP.md``.
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
    """Concrete ticker list for an API provider, from explicit list or preset.

    When ``data.membership`` is set (point-in-time mode) the universe is the
    **union of every name that was ever a constituent** in ``[start, end]`` — so a
    name that later left the index still has price history loaded for the period it
    was a member.  The panel is then masked per-bar to each date's actual
    constituents at load time (see :mod:`quant_fund_agent.data.membership`).
    """
    if data.membership:
        from quant_fund_agent.data.membership import union_members
        symbols = union_members(data.start, data.end, index=data.membership)
    elif data.tickers:
        symbols = [t.strip().upper() for t in data.tickers if t.strip()]
    elif data.universe_preset:
        symbols = load_preset(data.universe_preset)
    else:
        raise ValueError(
            "No universe configured: set data.membership, data.tickers or "
            f"data.universe_preset (presets: {available_presets()})."
        )
    if data.n_tickers is not None and len(symbols) > data.n_tickers:
        # NB: capping the PIT union trades survivorship completeness for speed
        # (front-of-list = alphabetical); leave n_tickers unset for a full run.
        symbols = symbols[: data.n_tickers]
    return symbols
