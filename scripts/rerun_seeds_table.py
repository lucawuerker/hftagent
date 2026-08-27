#!/usr/bin/env python
"""Multi-seed replication table for chapter arms 1 (LDU8), 4 (L1H), 6 (L4WF).

One row per run (the original s0, the s0b replication where it exists, and the
seed-1..4 reruns launched 2026-08-21/22, target 5 runs per arm), same columns/definitions as
scripts/alpha_arms_table.py, plus a per-arm mean / sd / n summary.  Runs whose
local post-analysis is missing fall back to the master table row
(data/comparisons/thesis_ablation/tables/master_table.csv) when one exists.

Output: data/comparisons/thesis_rerun_seeds/{runs.csv, runs.tex, arm_summary.csv}.
Re-run any time; rows appear as the post-analysis watcher finishes them.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
MASTER = ROOT / "data/comparisons/thesis_ablation/tables/master_table.csv"
OUT = ROOT / "data/comparisons/thesis_rerun_seeds"

spec = importlib.util.spec_from_file_location("aat", ROOT / "scripts/alpha_arms_table.py")
aat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aat)

# arm -> [(run, curated-book race label, display)]
RUNS = {
    "1 LDU8": [("LDU8_terra_s0", "LDU8CUR_terra_s0", "s0"),
               ("LDU8_terra_s1", "LDU8_terra_s1CUR", "s1"),
               ("LDU8_terra_s2", "LDU8_terra_s2CUR", "s2"),
               ("LDU8_terra_s3", "LDU8_terra_s3CUR", "s3"),
               ("LDU8_terra_s4", "LDU8_terra_s4CUR", "s4")],
    "4 L1H": [("L1H_terra_s0", "L1HCUR_terra_s0", "s0"),
              ("L1H_terra_s0b", "L1HCUR_terra_s0b", "s0b"),
              ("L1H_terra_s1", "L1H_terra_s1CUR", "s1"),
              ("L1H_terra_s2", "L1H_terra_s2CUR", "s2"),
              ("L1H_terra_s3", "L1H_terra_s3CUR", "s3")],
    # evolved arm: the book race IS the lineage-replay snapshot race (label = run)
    "6 L4WF": [("L4WF_terra_s0", "L4WF_terra_s0", "s0"),
               ("L4WF_terra_s1", "L4WF_terra_s1", "s1"),
               ("L4WF_terra_s2", "L4WF_terra_s2", "s2"),
               ("L4WF_terra_s3", "L4WF_terra_s3", "s3"),
               ("L4WF_terra_s4", "L4WF_terra_s4", "s4")],
}
COLS = ["preq_mean", "preq_se", "preq_hit", "lasso_book_mean", "lasso_book_se",
        "lasso_book_hit", "lasso_pool_mean", "med_abs_ic", "flip_share",
        "n_book", "n_eff", "mean_abs_corr", "n_trials", "cost_usd"]


def master_row(run: str) -> dict | None:
    if not MASTER.exists():
        return None
    m = pd.read_csv(MASTER, dtype={"arm": str})
    r = m[m["run"] == run]
    if r.empty:
        return None
    r = r.iloc[0].to_dict()
    r["source"] = "master_table"
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, runs in RUNS.items():
        for run, cur, disp in runs:
            if (ANA / run / "prequential_record.csv").exists() and (ANA / run / "diversity.json").exists():
                r = aat.row_for(run, cur, f"{arm} {disp}")
                r["source"] = "local_analysis"
                if not (PIT / f"{cur}_summary.csv").exists():
                    r["lasso_book_mean"] = r["lasso_book_se"] = r["lasso_book_hit"] = math.nan
                # evolved arm: the plain-label race is the book; pool = <run>POOL
                if arm.endswith("L4WF"):
                    pm, pse, ph = aat._pit(f"{run}POOL")
                    r["lasso_pool_mean"], r["lasso_pool_se"], r["lasso_pool_hit"] = pm, pse, ph
            else:
                r = master_row(run)
                if r is None:
                    continue
                r["arm"] = f"{arm} {disp}"
            r["arm_group"] = arm
            r["run"] = run
            rows.append(r)
    if not rows:
        print("no runs analysed yet")
        return
    df = pd.DataFrame(rows)
    keep = ["arm_group", "arm", "run", "source"] + [c for c in COLS if c in df.columns]
    df = df[keep]
    df.to_csv(OUT / "runs.csv", index=False)

    summ = []
    for g, d in df.groupby("arm_group", sort=False):
        rec = {"arm_group": g, "n_runs": len(d)}
        for c in ["preq_mean", "lasso_book_mean", "lasso_pool_mean", "med_abs_ic",
                  "flip_share", "n_book", "n_eff", "mean_abs_corr", "cost_usd"]:
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            rec[f"{c}_mean"] = float(v.mean()) if len(v) else math.nan
            rec[f"{c}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else math.nan
            rec[f"{c}_n"] = int(len(v))
        summ.append(rec)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT / "arm_summary.csv", index=False)

    tex = aat.latex(df.fillna({"lasso_book_mean": np.nan}))
    (OUT / "runs.tex").write_text(tex + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.drop(columns=["arm_group"]).round(4).to_string(index=False))
    print()
    print(sdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
