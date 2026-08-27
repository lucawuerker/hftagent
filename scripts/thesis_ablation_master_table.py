"""Assemble the single master results table of the ablation chapter.

Reads data/comparisons/thesis_ablation/tables/{ladder_summary,per_factor_all}.csv
and, if present, the 101-alpha zoo benchmark row from
data/comparisons/wf_arm_analysis_local/zoo/ + .../pit_combiners/zoo_summary.csv.

Writes tables/tex/ladder_master.tex and tables/master_table.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "data/comparisons/thesis_ablation/tables"
ANA = REPO / "data/comparisons/wf_arm_analysis_local"

ARM_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8a", "8b", "9", "MQ", "MQc", "Z"]
BLOCKRULE_AFTER = {"4", "7", "8b", "9", "MQc"}


def median_abs_ic() -> dict[str, tuple[float, int]]:
    df = pd.read_csv(TAB / "per_factor_all.csv")
    out = {}
    for arm, d in df.groupby("arm", sort=False):
        ok = d.dropna(subset=["ic_fit", "ic_wf_blockmean"])
        ok = ok[ok["ic_fit"] != 0]
        if len(ok):
            out[str(arm)] = (float(ok["ic_wf_blockmean"].abs().median()), len(ok))
    return out


def fixed_book_row(arm: str, label: str, tag: str,
                   n_trials=np.nan, cost=0.0) -> dict | None:
    """The 101 formulaic alphas benchmark, if it has been computed.

    The zoo has no run and therefore no prequential record of its own.  Its
    book is fixed, so the PIT race's LightGBM column -- the same gradient
    boosting combiner refitted on everything before each block -- IS the
    exact analogue of the arms' prequential record, and is reported in that
    column (flagged in the caption).
    """
    pf = ANA / arm / "per_factor_blocks.csv"
    div = ANA / arm / "diversity.json"
    pit = ANA / "pit_combiners" / f"{arm}_summary.csv"
    jsl = ANA / "pit_combiners" / f"{arm}.jsonl"
    if not pf.exists():
        return None
    d = pd.read_csv(pf)
    ok = d.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    row = {
        "arm": tag, "name": label, "run": arm,
        "med_abs_ic": float(ok["ic_wf_blockmean"].abs().median()),
        "flip_share": float((np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()),
        "flip_n": int(len(ok)),
        "n_book": int(len(d)),
        "n_trials": n_trials, "cost_usd": cost,
    }
    if div.exists():
        j = json.loads(div.read_text())
        row["n_eff"] = j.get("effective_n_participation_ratio")
        row["mean_abs_corr"] = j.get("mean_abs_corr")
    if pit.exists():
        p = pd.read_csv(pit)
        for method, pre in (("lasso", "lasso_book"), ("lightgbm", "preq")):
            m = p[p["method"] == method]
            if not len(m):
                continue
            r = m.iloc[0]
            n = float(r["n_blocks"])
            row[f"{pre}_mean"] = float(r["blockmean"])
            row[f"{pre}_se"] = float(r["blockstd"]) / np.sqrt(n)
            row[f"{pre}_hit"] = float(r["hit"])
            row[f"{pre}_blocks"] = n
        row["lasso_book_n_avail"] = float(row["n_book"])
        best = p.loc[p["blockmean"].idxmax()]
        row["pit_best_book_mean"] = float(best["blockmean"])
        row["pit_best_book_method"] = str(best["method"])
    if jsl.exists():
        sel = [json.loads(l) for l in jsl.read_text().splitlines()]
        sel = [r["diag"].get("n_nonzero_coef") for r in sel
               if r["method"] == "lasso" and isinstance(r.get("diag"), dict)]
        sel = [x for x in sel if x is not None]
        if sel:
            row["lasso_book_n_selected"] = float(np.mean(sel))
    return row


def f(x, nd=4, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{nd}f}"


def main() -> None:
    lad = pd.read_csv(TAB / "ladder_summary.csv", dtype={"arm": str})
    med = median_abs_ic()
    lad["med_abs_ic"] = lad["arm"].map(lambda a: med.get(a, (np.nan, 0))[0])
    extra = [
        fixed_book_row("L1HBNOLA_4omini_s0", "L1HB-4o-mini (clean)", "MQc",
                       n_trials=113, cost=0.33),
        fixed_book_row("zoo", "101 alphas", "Z"),
    ]
    extra = [r for r in extra if r is not None]
    if extra:
        lad = pd.concat([lad, pd.DataFrame(extra)], ignore_index=True)
    lad["_o"] = lad["arm"].map(lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
    lad = lad.sort_values("_o")

    # selection share of the book actually used by the Lasso
    share = lad["lasso_book_n_selected"] / lad["lasso_book_n_avail"]

    lines = [
        r"\begin{tabular}{|c|l|rrc|rrc|rr|rrr|rr|}",
        r"\hline",
        r"\multirow{2}{*}{\#} & \multirow{2}{*}{Arm} & "
        r"\multicolumn{3}{c|}{Prequential (GBM)} & "
        r"\multicolumn{3}{c|}{Walk-forward (Lasso)} & "
        r"\multicolumn{2}{c|}{Per factor} & "
        r"\multicolumn{3}{c|}{Independence} & "
        r"\multicolumn{2}{c|}{Cost} \\",
        r" & & $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & $H$ "
        r"& $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & $H$ "
        r"& $\widetilde{|\overline{\mathrm{IC}}|}$ & $\Phi$ "
        r"& $|\mathcal B|$ & $N_{\mathrm{eff}}$ & $\bar{|\rho|}$ "
        r"& $N_{\mathrm{tr}}$ & \$ \\",
        r"\hline",
    ]
    for i, (_, r) in enumerate(lad.iterrows()):
        sel = share.iloc[lad.index.get_loc(_)] if False else None
        cells = [
            str(r["arm"]),
            r"\texttt{" + str(r["name"]).replace("_", r"\_") + "}",
            f(r.get("preq_mean")), f(r.get("preq_se")),
            f(r.get("preq_hit"), 1),
            f(r.get("lasso_book_mean")), f(r.get("lasso_book_se")),
            f(r.get("lasso_book_hit"), 1),
            f(r.get("med_abs_ic")), f(r.get("flip_share"), 2),
            f(r.get("n_book"), 0), f(r.get("n_eff"), 1), f(r.get("mean_abs_corr"), 2),
            f(r.get("n_trials"), 0), f(r.get("cost_usd"), 2),
        ]
        lines.append(" & ".join(cells) + r" \\")
        if str(r["arm"]) in BLOCKRULE_AFTER:
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}"]
    out = TAB / "tex/ladder_master.tex"
    out.write_text("\n".join(lines) + "\n")

    keep = ["arm", "name", "run", "preq_mean", "preq_se", "preq_hit",
            "lasso_book_mean", "lasso_book_se", "lasso_book_hit",
            "lasso_book_n_avail", "lasso_book_n_selected",
            "lasso_pool_mean", "lasso_pool_n_avail", "lasso_pool_n_selected",
            "med_abs_ic", "flip_share", "n_book", "n_pool",
            "n_eff", "mean_abs_corr", "n_trials", "cost_usd",
            "pit_best_book_mean", "pit_best_book_method"]
    lad.reindex(columns=keep).to_csv(TAB / "master_table.csv", index=False)
    print(out)
    print(lad.reindex(columns=keep).to_string(index=False))


if __name__ == "__main__":
    main()
