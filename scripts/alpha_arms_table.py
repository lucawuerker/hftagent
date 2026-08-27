#!/usr/bin/env python
"""Comparison-table rows for the 2026-08-19 overnight arms.

Emits exactly the columns of the thesis comparison table

    Arm | Prequential (GBM): IC, s/sqrt(10), H
        | Walk-forward (Lasso): IC, s/sqrt(10), H
        | Per factor: median |mean block IC|, Phi
        | Independence: |B|, N_eff, mean|rho|
        | Cost: N_trials, $

with the ladder's definitions, identical to scripts/thesis_final_figures.py
(_s0b_row) and scripts/thesis_ablation_master_table.py:

  * prequential  — the run's own combined OOS IC on the 10 phase-2 walk-forward
    blocks (generations 11-20, LightGBM refit), from prequential_record.csv;
    H = share of positive blocks.
  * walk-forward — LassoCV in the point-in-time combiner race on the CURATED
    book (<ARM>CUR label), mean of the 10 per-block ICs.
  * per factor   — median |mean block IC| and sign-flip share Phi over the book
    members with finite, non-zero fit-window IC (so both agree on one subset).
  * independence — book size, participation-ratio effective N, mean |rho|.
  * cost         — billed N_trials from the manifest and metered LLM spend.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
PRERUNS = ROOT / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def _pit(label: str, method: str = "lasso") -> tuple[float, float, float]:
    p = PIT / f"{label}_summary.csv"
    if not p.exists():
        return (math.nan,) * 3
    df = pd.read_csv(p)
    row = df[df["method"] == method]
    if row.empty:
        return (math.nan,) * 3
    r = row.iloc[0]
    return (float(r["blockmean"]),
            float(r["blockstd"]) / math.sqrt(float(r["n_blocks"])),
            float(r["hit"]))


def row_for(run: str, cur_label: str | None, name: str | None = None) -> dict:
    d = ANA / run
    preq = pd.read_csv(d / "prequential_record.csv")
    ics = preq[preq["generation"] >= 11]["combined_oos_ic"].dropna().to_numpy()
    pf = pd.read_csv(d / "per_factor_blocks.csv")
    ok = pf.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    div = json.loads((d / "diversity.json").read_text())

    man_p = PRERUNS / run / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    usage_p = PRERUNS / run / "evolution/llm_usage.json"
    if usage_p.exists():
        usage = json.loads(usage_p.read_text())
    else:
        usage = man.get("llm_usage", {})
    cost = sum(v.get("cost_usd", 0.0) for v in usage.get("by_role", {}).values())

    book_m, book_se, book_h = _pit(cur_label) if cur_label else (math.nan,) * 3
    pool_m, pool_se, pool_h = _pit(run)
    return {
        "arm": name or run,
        "run": run,
        "preq_mean": float(np.mean(ics)) if len(ics) else math.nan,
        "preq_se": (float(np.std(ics, ddof=1)) / math.sqrt(len(ics))
                    if len(ics) > 1 else math.nan),
        "preq_hit": float((ics > 0).mean()) if len(ics) else math.nan,
        "preq_blocks": int(len(ics)),
        "lasso_book_mean": book_m, "lasso_book_se": book_se,
        "lasso_book_hit": book_h,
        "lasso_pool_mean": pool_m, "lasso_pool_se": pool_se,
        "lasso_pool_hit": pool_h,
        "med_abs_ic": float(ok["ic_wf_blockmean"].abs().median()),
        "flip_share": float(
            (np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()),
        "n_book": int(div["n_factors"]),
        "n_eff": float(div["effective_n_participation_ratio"]),
        "mean_abs_corr": float(div["mean_abs_corr"]),
        "n_trials": man.get("n_trials", float("nan")),
        "cost_usd": cost,
    }


def latex(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(
            f"{r['arm']} & {r['preq_mean']:.4f} & {r['preq_se']:.4f} & "
            f"{r['preq_hit']*10:.0f}/10 & "
            f"{r['lasso_book_mean']:.4f} & {r['lasso_book_se']:.4f} & "
            f"{r['lasso_book_hit']*10:.0f}/10 & "
            f"{r['med_abs_ic']:.4f} & {r['flip_share']:.2f} & "
            f"{r['n_book']:.0f} & {r['n_eff']:.1f} & {r['mean_abs_corr']:.3f} & "
            f"{r['n_trials']:.0f} & {r['cost_usd']:.2f} \\\\")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="run[:CURLABEL[:display name]] triples")
    ap.add_argument("--out", default="data/comparisons/alpha_arms/table.csv")
    args = ap.parse_args()

    rows = []
    for spec in args.arms:
        parts = spec.split(":")
        run = parts[0]
        cur = parts[1] if len(parts) > 1 and parts[1] else None
        name = parts[2] if len(parts) > 2 else None
        if not (ANA / run / "prequential_record.csv").exists():
            print(f"[skip] {run}: no analysis yet")
            continue
        rows.append(row_for(run, cur, name))
    if not rows:
        print("nothing to report")
        return
    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 40)
    print(df.to_string(index=False))
    print("\n% LaTeX rows\n" + latex(df))
    (out.with_suffix(".tex")).write_text(latex(df) + "\n")
    print(f"\nwrote {out} and {out.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
