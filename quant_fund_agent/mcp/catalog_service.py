"""Factor-catalog logic shared by the quant-catalog MCP server and its fallback.

Reads the persisted factor database (``data/factors/factor_db.json``) and builds
the compact per-factor summary (id, name, category, description, IC/ICIR at the
1 / 6 / 60-bar horizons) that the Selector LLM reasons over.  The server is a
thin protocol wrapper over :func:`load_factor_catalog`; the in-process fallback
calls it directly, so both paths return identical JSON.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Default location, captured at import for backward-compatible importers (e.g.
# the ``HAS_DB`` guard in tests).  The live lookup below is what actually drives
# ``load_factor_catalog`` so per-run redirection still works.
FACTOR_DB_PATH = Path(os.getenv("FACTOR_DB_PATH", "data/factors/factor_db.json"))


def _factor_db_path() -> Path:
    """Factor DB the Selector reads — resolved at call time.

    Read live from ``FACTOR_DB_PATH`` on every call so a walk-forward backtest
    can point the Selector at its run-scoped factor DB (seeded per the run's
    ``factor_universe``) without re-importing this module.
    """
    return Path(os.getenv("FACTOR_DB_PATH", "data/factors/factor_db.json"))


def load_factor_catalog() -> list[dict[str, Any]]:
    """Return a compact JSON-serialisable summary of every persisted factor."""
    path = _factor_db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Factor DB not found at {path}. Run `python run_all_factors.py` first."
        )

    raw = json.loads(path.read_text())
    catalog: list[dict[str, Any]] = []

    for f in raw.get("factors", []):
        bm = f.get("backtest_metrics") or {}
        by_h = bm.get("ic_by_horizon", {})

        catalog.append({
            "factor_id": f["id"],
            "name": f["name"],
            "category": f.get("category", ""),
            "description": f.get("description", ""),
            "ic_1": (by_h.get("1") or {}).get("ic"),
            "ic_6": (by_h.get("6") or {}).get("ic"),
            "ic_60": (by_h.get("60") or {}).get("ic"),
            "icir_1": (by_h.get("1") or {}).get("ic_ir"),
            "icir_6": (by_h.get("6") or {}).get("ic_ir"),
            "icir_60": (by_h.get("60") or {}).get("ic_ir"),
        })

    return catalog
