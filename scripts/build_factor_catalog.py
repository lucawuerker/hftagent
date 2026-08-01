"""Build the product factor catalog from evolution runs' kept-pools.

Heuristic (agreed 2026-07-29, docs/research-evolution/FACTOR_CATALOG_HEURISTIC.md):
Stage 0 eligibility -> Stage 1 canonical-AST clone-family dedup (PRIMARY)
-> Stage 2 within-bucket correlation pruning (|rho| >= --corr, keep the top
TWO members per high-correlation cluster, kill only beyond that). Generous by
design; no new model fits (zero extra trials); the catalog records the global
n_trials of its contributing runs for downstream deflation.

Usage::

    QF_USE_MCP=0 ./venv/bin/python scripts/build_factor_catalog.py \
        --config quant.config.nasdaq100_2010.yaml \
        --preruns L2_opus5_s0 --name nasdaq100_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_factor_catalog")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--preruns", required=True,
                   help="Comma list of prerun names whose kept_pools feed the catalog.")
    p.add_argument("--name", required=True, help="Catalog name under data/books/catalog_<name>/")
    p.add_argument("--ast-sim", type=float, default=0.95,
                   help="AST similarity at/above which two factors are clones (default 0.95).")
    p.add_argument("--corr", type=float, default=0.85,
                   help="Within-bucket |correlation| for the keep-two rule (default 0.85).")
    p.add_argument("--min-coverage", type=float, default=0.5)
    args = p.parse_args()

    os.environ["QF_CONFIG_FILE"] = args.config
    os.environ.setdefault("QF_USE_MCP", "0")

    import numpy as np
    import pandas as pd

    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.ast_novelty import ast_subtree_similarity

    cfg_name = default_config_name(get_settings().data)

    # ── collect candidates from every prerun's kept_pool ──
    candidates: list[dict] = []
    n_trials_global = 0
    for prerun in [s.strip() for s in args.preruns.split(",") if s.strip()]:
        state_path = (REPO / "data" / "workspaces" / cfg_name / "preruns"
                      / prerun / "evolution" / "state.json")
        st = json.loads(state_path.read_text())
        n_trials_global += int(st.get("n_trials", 0))
        for eg in st.get("kept_pool", []):
            g, f = eg["genome"], eg["fitness"]
            prog = g["programs"][0]
            d = f.get("diagnostics", {}) or {}
            obj = f.get("objective", {}) or {}
            candidates.append({
                "factor_id": prog["factor_id"],
                "code": prog["code"],
                "category": str(prog.get("category") or "other"),
                "mechanism": str(prog.get("mechanism") or ""),
                "expected_sign": prog.get("expected_sign"),
                "prediction_horizon": prog.get("prediction_horizon"),
                "trading_idea": prog.get("trading_idea", ""),
                "name": prog.get("name", ""),
                "prerun": prerun,
                "generation": g.get("generation"),
                "operator": g.get("operator"),
                "score": obj.get("marginal_value"),
                "coverage": d.get("coverage"),
                "degradation_ratio": d.get("degradation_ratio"),
                "is_ic": d.get("is_ic"),
            })
    log.info("collected %d kept-pool candidates from %s (global n_trials=%d)",
             len(candidates), args.preruns, n_trials_global)

    # ── Stage 0: eligibility ──
    fields = set(usable_fields())
    eligible: list[dict] = []
    seen_ids: set[str] = set()
    for c in candidates:
        if c["factor_id"] in seen_ids:      # same id across checkpoints → newest wins
            continue
        if c["score"] is None or not np.isfinite(c["score"]):
            continue
        if c["coverage"] is not None and c["coverage"] < args.min_coverage:
            continue
        seen_ids.add(c["factor_id"])
        eligible.append(c)
    log.info("stage 0 eligibility: %d / %d survive", len(eligible), len(candidates))

    # ── Stage 1: canonical-AST clone families (PRIMARY dedup) ──
    # Greedy family assignment in descending score: a candidate joins the first
    # existing family whose representative it matches at >= ast_sim; otherwise
    # it founds a new family. The representative (best score) survives.
    eligible.sort(key=lambda c: -(c["score"] or 0.0))
    reps: list[dict] = []
    for c in eligible:
        family = None
        for r in reps:
            try:
                if ast_subtree_similarity(c["code"], r["code"]) >= args.ast_sim:
                    family = r
                    break
            except Exception:  # noqa: BLE001 — unparsable code never blocks
                continue
        if family is None:
            c["clone_family"] = c["factor_id"]
            reps.append(c)
        else:
            family.setdefault("clones", []).append(c["factor_id"])
    log.info("stage 1 AST dedup: %d clone-family representatives", len(reps))

    # ── Stage 2: within-bucket correlation, keep-two-per-cluster ──
    panel = svc._load_panel_cached("ticker_data", sorted(fields), n_tickers=None)
    close = panel["close"]
    sigs: dict[str, pd.DataFrame] = {}
    kept: list[dict] = []
    dropped_corr = 0
    for c in reps:                       # already sorted by score desc
        try:
            cls = compile_factor(c["code"], c["factor_id"])
            s = compute_signal(cls, panel).reindex(index=close.index,
                                                   columns=close.columns)
        except Exception as e:  # noqa: BLE001
            log.info("skip %s (signal failed: %s)", c["factor_id"], e)
            continue
        arr = s.to_numpy(dtype=float).ravel()
        if not np.isfinite(arr).any() or np.nanstd(arr) == 0:
            continue
        bucket = (c["category"], c["mechanism"] or None)
        high_corr_hits = []
        for k in kept:
            if (k["category"], k["mechanism"] or None) != bucket:
                continue
            a, b = sigs[k["factor_id"]], arr
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 500:
                continue
            rho = np.corrcoef(a[m], b[m])[0, 1]
            if np.isfinite(rho) and abs(rho) >= args.corr:
                high_corr_hits.append((k["factor_id"], float(rho)))
        if len(high_corr_hits) >= 2:     # two members of this cluster already kept
            dropped_corr += 1
            continue
        c["bucket"] = list(bucket)
        c["corr_neighbours"] = high_corr_hits
        sigs[c["factor_id"]] = arr
        kept.append(c)
    log.info("stage 2 corr keep-two: kept %d (dropped %d)", len(kept), dropped_corr)

    out_dir = REPO / "data" / "books" / f"catalog_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        "name": args.name,
        "config": cfg_name,
        "preruns": args.preruns.split(","),
        "n_trials_global": n_trials_global,
        "params": {"ast_sim": args.ast_sim, "corr": args.corr,
                   "min_coverage": args.min_coverage},
        "n_factors": len(kept),
        "factors": [{k: v for k, v in c.items() if k != "code"} for c in kept],
    }
    (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, default=str))
    import collections
    cats = collections.Counter(c["category"] for c in kept)
    print(f"\ncatalog '{args.name}': {len(kept)} factors "
          f"(from {len(candidates)} candidates; n_trials_global={n_trials_global})")
    print("by category:", dict(cats))
    print("->", out_dir / "catalog.json")


if __name__ == "__main__":
    main()
