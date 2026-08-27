#!/usr/bin/env python
"""Thesis-format LaTeX rows for the multi-seed replication (arms 1/4/6).

Two tables, both in the chapter's 15-column layout (Prequential LightGBM:
IC, s/sqrt(10), IC_defl, H | Walk-forward Lasso: same | per factor: med|IC|, Phi
| independence: |B|, N_eff, mean|rho| | cost: N_tr, $) plus an extra cost
column "$ adj." = the metered spend minus the cost of the oversized
existing-id list (see below):

  * runs.tex      one row per run
  * means.tex     one row per arm = mean over its three runs

Definitions follow scripts/thesis_ablation_master_table.py.  IC_defl subtracts
the luck term sqrt(2 ln N_tr / n_obs) with n_obs = 127,911 (the pooled
observation count of the walk-forward record, identical across arms).

Arm 1 (LDU8): every book is dominated by level-class signals (median per-name
lag-1 autocorrelation rho > 0.9: price levels / quarterly-stepped ratios whose
block-centred IC is inflated — the GP finding), so its rows are computed on
the LEVEL-CLEAN subset (rho < 0.9) of the curated book; where the clean book is
empty (s2) the clean kept pool is used instead (flagged).  A run's own
prequential record cannot be purged of those members, so the prequential
column of the LDU8 rows is the PIT LightGBM race on the clean subset (flagged).

Cost split: from 2026-08-16/17 the shared researcher package holds ~7.1k
factors and their id list (~61k tokens) was spliced into every brainstorm /
mutation / crossover prompt (L4WF_s0-era list: ~600 ids, ~5k tokens).  For the
affected runs "$ adj." removes the estimated extra input tokens:
n_calls(role) = input_tokens / (baseline_per_call + delta), extra cost =
n_calls * delta * $2.5/M, with baselines from the L4WF_s0 transcript
(brainstorm 10.8k, mutation 13.3k, crossover 14.9k; LDU8 brainstorm 4.3k
without papers) and delta = 61k - 5k = 56k tokens.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))
ANA = ROOT / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
WS = ROOT / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
TAB = ROOT / "data/comparisons/thesis_ablation/tables"
OUT = ROOT / "data/comparisons/thesis_rerun_seeds"
LEVEL = ROOT / "data/comparisons/wf_book_analysis/derived/pool_level_profiles.csv"
N_OBS = 127_911
PRICE_IN = 2.5e-6          # $ per input token, gpt-5.6-terra
DELTA_TOKENS = 56_000      # extra id-list tokens per affected call
BASELINE = {"brainstorm": 10_800, "mutation": 13_300, "crossover": 14_900,
            "hypothesis": 13_300}
BASELINE_NOPAPER_BRAINSTORM = 4_300
# runs whose prompts carried the 7k-id list (package grew 2026-08-16/17)
AFFECTED = {"LDU8_terra_s1", "LDU8_terra_s2", "L1H_terra_s0b", "L1H_terra_s1",
            "L4WF_terra_s1", "L4WF_terra_s2"}

spec = importlib.util.spec_from_file_location("ub", ROOT / "scripts/thesis_union_books.py")
ub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ub)

ARMS = {
    "1": ("LDU8", [("LDU8_terra_s0", "s0"), ("LDU8_terra_s1", "s1"), ("LDU8_terra_s2", "s2")]),
    "4": ("L1H", [("L1H_terra_s0", "s0"), ("L1H_terra_s0b", "s0b"), ("L1H_terra_s1", "s1")]),
    "6": ("L4WF", [("L4WF_terra_s0", "s0"), ("L4WF_terra_s1", "s1"), ("L4WF_terra_s2", "s2")]),
}
BOOK_LABEL = {  # curated-book PIT race label per run
    "LDU8_terra_s0": "LDU8CUR_terra_s0", "L1H_terra_s0": "L1HCUR_terra_s0",
    "L1H_terra_s0b": "L1HCUR_terra_s0b", "L1H_terra_s1": "L1H_terra_s1CUR",
    "L4WF_terra_s0": "L4WF_terra_s0", "L4WF_terra_s1": "L4WF_terra_s1",
    "L4WF_terra_s2": "L4WF_terra_s2",
}


def luck(n_trials: float) -> float:
    return math.sqrt(2.0 * math.log(n_trials)) / math.sqrt(N_OBS) if n_trials > 1 else 0.0


def pit(label: str, method: str) -> tuple[float, float, float]:
    p = PIT / f"{label}_summary.csv"
    if not p.exists():
        return (math.nan,) * 3
    df = pd.read_csv(p)
    r = df[df["method"] == method]
    if r.empty:
        return (math.nan,) * 3
    r = r.iloc[0]
    return float(r["blockmean"]), float(r["blockstd"]) / math.sqrt(float(r["n_blocks"])), float(r["hit"])


def pool_per_factor(run: str, fids: set[str]) -> pd.DataFrame:
    """ic_fit + mean WF block IC for pool members outside the book analysis
    (same _pooled_ic / block convention as wf_arm_factor_analysis; cached)."""
    cache = OUT / f"{run}_pool_per_factor.csv"
    have = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["factor_id"])
    todo = sorted(fids - set(have["factor_id"]))
    if not todo:
        return have
    import os, sys
    os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
    os.environ.setdefault("QF_USE_MCP", "0")
    sys.path.insert(0, str(ROOT / "scripts"))
    from wf_common import load_or_compute_signal
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import _pooled_ic
    spec_a = importlib.util.spec_from_file_location("wfa", ROOT / "scripts/wf_arm_factor_analysis.py")
    wfa = importlib.util.module_from_spec(spec_a)
    spec_a.loader.exec_module(wfa)
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()), n_tickers=None)
    close = panel["close"]; idx = close.index
    fit_mask = np.asarray(idx < pd.Timestamp("2021-07-20"))
    blocks = wfa.block_windows(run, idx)
    discover_factors()
    st = json.loads((WS / run / "evolution/state.json").read_text())
    code = {}
    for eg in st.get("kept_pool", []):
        for prog in eg["genome"]["programs"]:
            code[prog["factor_id"]] = prog["code"]
    code.update(ub.final_book(run))
    rows = []
    for fid in todo:
        sig = load_or_compute_signal(fid, code[fid], panel, idx, close.columns).astype(float)
        rec = {"factor_id": fid,
               "ic_fit": _pooled_ic(sig, close, 6, row_mask=fit_mask, available_mask=fit_mask)[0]}
        bl = [ic for _, m, *_ in blocks
              if (ic := _pooled_ic(sig, close, 6, row_mask=m, available_mask=m)[0]) is not None]
        rec["ic_wf_blockmean"] = float(np.mean(bl)) if bl else None
        rows.append(rec)
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(cache, index=False)
    return out


def per_factor(run: str, fids: set[str] | None) -> tuple[float, float]:
    if (ANA / run / "per_factor_blocks.csv").exists():
        pf = pd.read_csv(ANA / run / "per_factor_blocks.csv")
    else:  # L4WF_s0: analysed on the server, rows in per_factor_all.csv (arm 6)
        pfa = pd.read_csv(TAB / "per_factor_all.csv", dtype={"arm": str})
        pf = pfa[pfa["arm"] == "6"]
    if fids is not None:
        missing = fids - set(pf["factor_id"])
        if missing:
            pf = pd.concat([pf, pool_per_factor(run, missing)], ignore_index=True)
        pf = pf[pf["factor_id"].isin(fids)]
    ok = pf.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    return (float(ok["ic_wf_blockmean"].abs().median()),
            float((np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()))


def usage(run: str) -> tuple[float, float, float]:
    """(n_trials, cost, cost_adj)."""
    man_p = WS / run / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    up = WS / run / "evolution/llm_usage.json"
    u = json.loads(up.read_text()) if up.exists() else man.get("llm_usage", {})
    by = u.get("by_role", {})
    cost = sum(v.get("cost_usd", 0.0) for v in by.values())
    extra = 0.0
    if run in AFFECTED:
        for role, v in by.items():
            if role not in BASELINE:
                continue
            base = BASELINE[role]
            if role == "brainstorm" and run.startswith("LDU8"):
                base = BASELINE_NOPAPER_BRAINSTORM
            n_calls = v["input_tokens"] / (base + DELTA_TOKENS)
            extra += n_calls * DELTA_TOKENS * PRICE_IN
    return float(man.get("n_trials", math.nan)), cost, cost - extra


def diversity_for(run: str, fids: set[str]) -> tuple[float, float]:
    book = ub.final_book(run)
    st_p = WS / run / "evolution/state.json"
    if st_p.exists():  # pool members live in the state (clean-pool fallback)
        st = json.loads(st_p.read_text())
        for eg in st.get("kept_pool", []):
            for prog in eg["genome"]["programs"]:
                book.setdefault(prog["factor_id"], prog["code"])
    members = {f: c for f, c in book.items() if f in fids}
    d = ub.diversity(members)
    return d["effective_n_participation_ratio"], d["mean_abs_corr"]


def row(run: str, arm_key: str, disp: str) -> dict:
    name = ARMS[arm_key][0]
    n_tr, cost, cost_adj = usage(run)
    L = luck(n_tr)
    rec = {"arm": arm_key, "run": run, "disp": disp, "n_trials": n_tr,
           "cost": cost, "cost_adj": cost_adj, "flag": ""}
    if name == "LDU8":
        lv = pd.read_csv(LEVEL).drop_duplicates(["arm", "factor_id"], keep="last")
        lv = lv[lv["arm"] == run]
        clean_book = json.loads((ROOT / f"data/comparisons/{run}_clean_book_fids.json").read_text())
        clean_pool = json.loads((ROOT / f"data/comparisons/{run}_clean_pool_fids.json").read_text())
        if len(clean_book) >= 2:
            fids, label, rec["flag"] = set(clean_book), f"{run}CLNBOOK", "clean book"
        else:
            fids, label, rec["flag"] = set(clean_pool), f"{run}CLNPOOL", "clean pool (clean book empty)"
        pm, pse, ph = pit(label, "lightgbm")
        rec["preq_src"] = "PIT lightgbm on clean subset"
        lm, lse, lh = pit(label, "lasso")
        med, phi = per_factor(run, fids)
        n_eff, rho = diversity_for(run, fids)
        n_book = len(fids)
    else:
        preq = pd.read_csv(ANA / run / "prequential_record.csv") if (ANA / run / "prequential_record.csv").exists() else None
        if preq is not None:
            ics = preq[preq["generation"] >= 11]["combined_oos_ic"].dropna().to_numpy()
            pm, pse, ph = float(ics.mean()), float(ics.std(ddof=1) / math.sqrt(len(ics))), float((ics > 0).mean())
        else:  # L4WF_s0 from the master table
            m = pd.read_csv(TAB / "master_table.csv", dtype={"arm": str})
            m = m[m["run"] == run].iloc[0]
            pm, pse, ph = float(m["preq_mean"]), float(m["preq_se"]), float(m["preq_hit"])
        rec["preq_src"] = "own prequential record"
        lm, lse, lh = pit(BOOK_LABEL[run], "lasso")
        if math.isnan(lm):  # L4WF_s0: book race ran on the server -> master table
            m = pd.read_csv(TAB / "master_table.csv", dtype={"arm": str})
            m = m[m["run"] == run].iloc[0]
            lm, lse, lh = float(m["lasso_book_mean"]), float(m["lasso_book_se"]), float(m["lasso_book_hit"])
        med, phi = per_factor(run, None)
        if (ANA / run / "diversity.json").exists():
            d = json.loads((ANA / run / "diversity.json").read_text())
            n_eff, rho, n_book = d["effective_n_participation_ratio"], d["mean_abs_corr"], d["n_factors"]
        else:
            m = pd.read_csv(TAB / "master_table.csv", dtype={"arm": str})
            m = m[m["run"] == run].iloc[0]
            n_eff, rho, n_book = float(m["n_eff"]), float(m["mean_abs_corr"]), int(m["n_book"])
    rec.update({"preq_mean": pm, "preq_se": pse, "preq_defl": max(pm - L, 0.0), "preq_hit": ph,
                "lasso_mean": lm, "lasso_se": lse, "lasso_defl": max(lm - L, 0.0), "lasso_hit": lh,
                "med_abs_ic": med, "flip_share": phi, "n_book": n_book, "n_eff": n_eff,
                "mean_abs_corr": rho})
    return rec


def tex_row(label: str, r: pd.Series, hit_fmt=lambda h: f"{h:.1f}") -> str:
    return (f"{label} & {r.preq_mean:.4f} & {r.preq_se:.4f} & {r.preq_defl:.4f} & {hit_fmt(r.preq_hit)} & "
            f"{r.lasso_mean:.4f} & {r.lasso_se:.4f} & {r.lasso_defl:.4f} & {hit_fmt(r.lasso_hit)} & "
            f"{r.med_abs_ic:.4f} & {r.flip_share:.2f} & {r.n_book:.0f} & {r.n_eff:.1f} & {r.mean_abs_corr:.2f} & "
            f"{r.n_trials:.0f} & {r.cost:.2f} & {r.cost_adj:.2f} \\\\")


HEADER = r"""\hline
Arm & \multicolumn{4}{c|}{Prequential (LightGBM)} & \multicolumn{4}{c|}{Walk-forward (Lasso)} & \multicolumn{2}{c|}{Per factor} & \multicolumn{3}{c|}{Independence} & \multicolumn{3}{c|}{Cost} \\
 & $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & $\mathrm{IC}_{\mathrm{defl}}$ & $H$ & $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & $\mathrm{IC}_{\mathrm{defl}}$ & $H$ & $\mathrm{med}\,|\overline{\mathit{IC}}|$ & $\Phi$ & $|\mathcal B|$ & $N_{\mathrm{eff}}$ & $\overline{|\rho|}$ & $N_{\mathrm{tr}}$ & \$ & \$ adj. \\
\hline"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row(run, k, disp) for k, (_, runs) in ARMS.items() for run, disp in runs]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "thesis_rows.csv", index=False)

    lines = [HEADER]
    for k, (name, _) in ARMS.items():
        for _, r in df[df["arm"] == k].iterrows():
            mark = r"$^{\dagger}$" if name == "LDU8" else ""
            mark += r"$^{\ddagger}$" if "pool" in r.flag else ""
            lines.append(tex_row(f"{k} ({r.disp}){mark}", r))
        lines.append(r"\hline")
    (OUT / "runs_thesis.tex").write_text("\n".join(lines) + "\n")

    cols = ["preq_mean", "preq_se", "preq_defl", "preq_hit", "lasso_mean", "lasso_se", "lasso_defl",
            "lasso_hit", "med_abs_ic", "flip_share", "n_book", "n_eff", "mean_abs_corr",
            "n_trials", "cost", "cost_adj"]
    means = df.groupby("arm")[cols].mean()
    sds = df.groupby("arm")[cols].std(ddof=1)
    means.to_csv(OUT / "thesis_means.csv")
    lines = [HEADER]
    for k, (name, _) in ARMS.items():
        m = means.loc[k]
        mark = r"$^{\dagger}$" if name == "LDU8" else ""
        lines.append(tex_row(f"{k}{mark} (mean of 3)", m, hit_fmt=lambda h: f"{h:.2f}"))
    lines.append(r"\hline")
    (OUT / "means_thesis.tex").write_text("\n".join(lines) + "\n")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.drop(columns=["preq_src"]).round(4).to_string(index=False))
    print("\n=== runs_thesis.tex ===\n" + (OUT / "runs_thesis.tex").read_text())
    print("=== means_thesis.tex ===\n" + (OUT / "means_thesis.tex").read_text())
    print("sd across runs:\n" + sds.round(4).to_string())


if __name__ == "__main__":
    main()
