#!/usr/bin/env python3
"""Build the 101-formulaic-alphas fixed book for the evolution runs.

Renders every registered seed alpha class into a self-contained, validator-passing
module string (injecting a literal ``prediction_horizon`` where the class relies on
the BaseFactor default) and writes a prebook JSON consumable by
``run_factor_evolution.py --fixed-book`` / ``--reference-book``.

Per FINAL_RUN_PLAN.md: the accepted book always *starts* from the 101 formulaic
alphas — every candidate's LOCO marginal value is measured against them (plus the
evolving archive), and structural novelty is measured against them via
``--reference-book``. Re-run this script whenever new seed alphas are added.

Usage:
    ./venv/bin/python scripts/build_formulaic_prebook.py \
        [--output data/prebooks/formulaic_101.json] [--horizon 6]
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/prebooks/formulaic_101.json")
    ap.add_argument("--horizon", type=int, default=6,
                    help="prediction_horizon injected when the class doesn't declare one")
    ap.add_argument("--all-seeds", action="store_true",
                    help="include every seed factor, not only the canonical alpha_NNN set")
    args = ap.parse_args()

    # importing the package triggers _discover, registering every seed alpha
    from quant_fund_agent.factors import registry  # noqa: PLC0415
    from quant_fund_agent.factors._discover import discover_factors  # noqa: PLC0415
    discover_factors()
    from quant_fund_agent.factors.inmem import compile_factor  # noqa: PLC0415
    from quant_fund_agent.research_eval.prebook import _ensure_prediction_horizon  # noqa: PLC0415

    classes = registry.get_all_factor_classes()
    members, failed = [], []
    for fid in sorted(classes):
        cls = classes[fid]
        import re
        module = inspect.getmodule(cls)
        if module is None or "factors.researcher" in (module.__name__ or ""):
            continue  # only seed/formulaic alphas belong in the standing book
        if not args.all_seeds and not re.fullmatch(r"alpha_\d+", fid):
            continue  # default: exactly the canonical 101 Kakushadze alphas
        try:
            code = inspect.getsource(module)
            horizon = getattr(cls, "prediction_horizon", None) or args.horizon
            code = _ensure_prediction_horizon(code, fid, int(horizon))
            compile_factor(code, fid)  # full validation, same as candidates
            members.append({"factor_id": fid, "code": code})
        except Exception as exc:  # noqa: BLE001 - report every failure
            failed.append({"factor_id": fid, "reason": str(exc)[:200]})

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kind": "formulaic_alphas",
        "members": members,
        "selected_factor_ids": [m["factor_id"] for m in members],
        "failed": failed,
    }, indent=1))
    print(f"wrote {out}: {len(members)} members, {len(failed)} failed")
    for f in failed:
        print("  FAILED", f["factor_id"], "-", f["reason"][:120])
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
