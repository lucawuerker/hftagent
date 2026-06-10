"""Phase 2 verification — factor gating by provider capability.

Run: PYTHONPATH=. ./venv/bin/python scripts/verify/verify_phase2_gating.py

Demonstrates, on the REAL committed factor DB, that:
  1. gating is a NO-OP on LOBSTER (every seed factor stays visible);
  2. a standard OHLCV provider gates out the microstructure factors;
  3. QF_FACTOR_GATING=0 restores the full catalog.
"""

import os
from collections import Counter

from quant_fund_agent.data.tiers import TIERS
from quant_fund_agent.mcp import catalog_service

LOBSTER_FIELDS = TIERS["standard"] | TIERS["microstructure"]
STANDARD_FIELDS = TIERS["standard"]


def catalog_ids():
    return [c["factor_id"] for c in catalog_service.load_factor_catalog()]


def main():
    full = catalog_ids()  # default provider = lobster (from config/env)
    total = len(full)
    print("===== 1. factor DB tier distribution (backfilled) =====")
    cat = catalog_service.load_factor_catalog()
    print("total factors:", total)
    print("by required_tier:", dict(Counter(c["required_tier"] for c in cat)))

    print("\n===== 2. LOBSTER provider (standard ∪ microstructure) — expect NO-OP =====")
    catalog_service._provider_available_fields = lambda: LOBSTER_FIELDS
    lob = catalog_ids()
    print(f"visible: {len(lob)}/{total}  (gating no-op on LOBSTER: {len(lob) == total})")

    print("\n===== 3. STANDARD OHLCV provider — expect microstructure gated out =====")
    catalog_service._provider_available_fields = lambda: STANDARD_FIELDS
    std = catalog_ids()
    gated = sorted(set(lob) - set(std))
    print(f"visible: {len(std)}/{total}   gated out: {len(gated)}")
    print("example gated factors:", gated[:10])

    print("\n===== 4. escape hatch QF_FACTOR_GATING=0 — expect full catalog =====")
    os.environ["QF_FACTOR_GATING"] = "0"
    off = catalog_ids()
    print(f"visible with gating OFF: {len(off)}/{total}  (restored: {len(off) == total})")
    del os.environ["QF_FACTOR_GATING"]

    print("\n===== assertions =====")
    assert len(lob) == total, "LOBSTER gating must be a no-op"
    assert len(std) < total and len(gated) > 0, "standard provider must gate microstructure"
    assert len(off) == total, "escape hatch must restore full catalog"
    print("ALL PHASE-2 GATING ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
