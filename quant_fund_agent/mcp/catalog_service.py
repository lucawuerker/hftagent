"""Factor-catalog logic shared by the quant-catalog MCP server and its fallback.

Reads the persisted factor database (``data/factors/factor_db.json``) and builds
the compact per-factor summary (id, name, category, description, IC/ICIR at the
1 / 6 / 60-bar horizons) that the Selector LLM reasons over.  The server is a
thin protocol wrapper over :func:`load_factor_catalog`; the in-process fallback
calls it directly, so both paths return identical JSON.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("mcp.catalog_service")

# Default location, captured at import for backward-compatible importers (e.g.
# the ``HAS_DB`` guard in tests).  The live lookup below is what actually drives
# ``load_factor_catalog`` so per-run redirection still works.
FACTOR_DB_PATH = Path(os.getenv("FACTOR_DB_PATH", "data/factors/factor_db.json"))


def _gating_enabled() -> bool:
    """Capability-gating is on unless ``QF_FACTOR_GATING=0`` (escape hatch)."""
    return os.getenv("QF_FACTOR_GATING", "1").strip().lower() not in ("0", "false", "no")


def _provider_available_fields() -> frozenset[str] | None:
    """Fields the configured provider supplies, or ``None`` if it can't resolve."""
    try:
        from quant_fund_agent.config import get_settings
        from quant_fund_agent.data import get_provider

        return frozenset(get_provider(get_settings()).available_fields())
    except Exception as e:  # provider/config trouble must never break the catalog
        log.warning("factor gating disabled — could not resolve provider fields: %s", e)
        return None


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
    records = raw.get("factors", [])

    # Populate the factor registry so required-field resolution is deterministic
    # in *both* the in-process and the (separate) MCP server processes — otherwise
    # the registry fallback would resolve differently across the boundary.
    try:
        from quant_fund_agent.factors import discover_factors

        discover_factors()
    except Exception:  # discovery is best-effort; stored metadata still works
        pass

    # Capability-gating: hide factors whose required fields the active provider
    # cannot supply (e.g. microstructure factors on a plain OHLCV vendor), so the
    # Selector never picks a factor that would later crash the architect.  This is
    # a no-op on LOBSTER (it supplies every field the seed library needs).
    from quant_fund_agent.data.tiers import required_tier, resolve_required_inputs

    available = _provider_available_fields() if _gating_enabled() else None
    if available is not None:
        from quant_fund_agent.data.tiers import is_compatible

        visible = [r for r in records if is_compatible(resolve_required_inputs(r), available)]
        if len(visible) != len(records):
            log.info("factor gating: %d/%d factors visible for provider fields=%d",
                     len(visible), len(records), len(available))
        records = visible

    catalog: list[dict[str, Any]] = []
    for f in records:
        bm = f.get("backtest_metrics") or {}
        by_h = bm.get("ic_by_horizon", {})

        catalog.append({
            "factor_id": f["id"],
            "name": f["name"],
            "category": f.get("category", ""),
            "description": f.get("description", ""),
            "required_tier": f.get("required_tier") or required_tier(resolve_required_inputs(f)),
            "ic_1": (by_h.get("1") or {}).get("ic"),
            "ic_6": (by_h.get("6") or {}).get("ic"),
            "ic_60": (by_h.get("60") or {}).get("ic"),
            "icir_1": (by_h.get("1") or {}).get("ic_ir"),
            "icir_6": (by_h.get("6") or {}).get("ic_ir"),
            "icir_60": (by_h.get("60") or {}).get("ic_ir"),
        })

    return catalog
