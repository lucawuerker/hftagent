"""FTSE 100 transfer-evaluation table (supervisor question 2026-08-23).

The already-published Nasdaq-100 factor books (arm 1 LDU8_terra_s0, arm 4
L1H_terra_s0b, arm 6 L4WF_terra_s0, benchmark zoo=101 formulaic alphas) are
re-scored - books frozen, nothing re-researched - on the FTSE 100 transfer
panel (quant.config.ftse100.yaml).  Column structure reproduces thesis
Table 4.2 verbatim; definitions follow scripts/thesis_ablation_master_table.py:

  * Prequential (LightGBM) column = the PIT LightGBM race.  On this panel NO
    search ran, so for ALL FOUR rows this is a fixed-book refit - the same
    convention the thesis uses for its 101-alpha row (its dagger footnote) -
    whereas the thesis column for arms 1/4/6 is the live run's own
    prequential record.
  * Walk-forward (Lasso) = the PIT LassoCV race.
  * IC_defl = research_eval.deflation.deflated_ic(|IC-bar|, n_obs, N_tr)
    ["deflated_ic"], with n_obs = the FTSE record's own pooled valid-pair
    count (recovered from the saved per-block prediction parquets via
    _pooled_ic - NOT the thesis's 127,911; the FTSE panel is smaller so the
    luck term is larger) and N_tr = the arm's trial count from the ORIGINAL
    Nasdaq search.  The zoo has no trial count -> IC_defl "--".
  * N_tr and $ carry over unchanged from the thesis run: they are properties
    of the search, which was paid for once on the Nasdaq panel; nothing was
    spent here.  (Arm 4 = the L1H_terra_s0b rerun: 86 trials, $14.79 from its
    own state.json/llm_usage.json - master_table.csv row 4 holds the OLDER
    L1H_terra_s0 run and is not used.)
  * med|IC-bar| and Phi on the FTSE per-factor block ICs, finite-and-nonzero
    ic_fit subset; |B|, N_eff, mean|rho| from the FTSE analysis (N_eff /
    mean|rho| thesis-identical definition; a pairwise-complete robustness
    version for BOTH panels is reported separately in REPORT.md).

Writes data/comparisons/ftse100_transfer/{transfer_table.csv,
transfer_table.tex,REPORT.md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "data/comparisons/ftse100_transfer"
ANA = OUT / "arm_analysis"
PIT = OUT / "pit_combiners"
DER = OUT / "derived"
MASTER = REPO / "data/comparisons/thesis_ablation/tables/master_table.csv"

ARMS = [
    # (tag, display name, prerun/arm, pit race label)
    ("1", "LDU8 (ungrounded ideation)", "LDU8_terra_s0", "LDU8CUR_ftse"),
    ("4", "L1H (graph+papers ideation)", "L1H_terra_s0b", "L1HCUR_ftse"),
    ("6", "L4WF (full system, evolution)", "L4WF_terra_s0", "L4WF_ftse"),
    ("Z", "101 formulaic alphas", "zoo", "zoo_ftse"),
]

# N_tr / $ of the ORIGINAL Nasdaq search (nothing was spent on the FTSE
# panel).  Arms 1/6 from master_table.csv; arm 4 = the L1H_terra_s0b rerun
# (state.json n_trials=86, llm_usage.json total $14.79); zoo: no search.
SEARCH_COST = {"1": (86, 6.41), "4": (86, 14.79), "6": (890, 98.44),
               "Z": (None, None)}

# Nasdaq-100 reference values.  Arm 4 = the L1H_terra_s0b RERUN (thesis
# figures' arm 4) - pinned, NOT read from master_table.csv which holds the
# older L1H_terra_s0.  preq_* is the run's own prequential record (arms
# 1/4/6) resp. the PIT LightGBM analogue (zoo).
ARM4_NASDAQ = {"preq_mean": 0.0491, "lasso_book_mean": 0.0653,
               "n_book": 22, "n_eff": 12.2, "med_abs_ic": 0.0112,
               "flip_share": 0.24, "mean_abs_corr": 0.10}


def nasdaq_ref(tag: str) -> dict:
    if tag == "4":
        return dict(ARM4_NASDAQ)
    m = pd.read_csv(MASTER, dtype={"arm": str})
    r = m[m["arm"] == tag]
    if not len(r):
        return {}
    r = r.iloc[0]
    return {k: r.get(k) for k in
            ("preq_mean", "lasso_book_mean", "n_book", "n_eff",
             "med_abs_ic", "flip_share", "mean_abs_corr")}


def ftse_row(tag: str, name: str, arm: str, label: str) -> dict:
    from quant_fund_agent.research_eval.deflation import deflated_ic

    row: dict = {"arm": tag, "name": name, "run": arm, "pit_label": label}
    n_trials, cost = SEARCH_COST[tag]
    row["n_trials"] = n_trials
    row["cost_usd"] = cost
    pf_path = ANA / arm / "per_factor_blocks.csv"
    if pf_path.exists():
        d = pd.read_csv(pf_path)
        ok = d.dropna(subset=["ic_fit", "ic_wf_blockmean"])
        ok = ok[ok["ic_fit"] != 0]
        row["n_computable"] = int(len(d))
        row["med_abs_ic"] = (float(ok["ic_wf_blockmean"].abs().median())
                             if len(ok) else np.nan)
        row["flip_share"] = (float((np.sign(ok["ic_fit"])
                                    != np.sign(ok["ic_wf_blockmean"])).mean())
                             if len(ok) else np.nan)
        row["flip_n"] = int(len(ok))
    div_path = ANA / arm / "diversity.json"
    if div_path.exists():
        j = json.loads(div_path.read_text())
        row["n_eff"] = j.get("effective_n_participation_ratio")
        row["mean_abs_corr"] = j.get("mean_abs_corr")
        row["n_failed"] = j.get("n_failed")
    n_obs = json.loads((DER / "n_obs.json").read_text()) \
        if (DER / "n_obs.json").exists() else {}
    pit_path = PIT / f"{label}_summary.csv"
    if pit_path.exists():
        p = pd.read_csv(pit_path)
        for method, pre in (("lightgbm", "preq"), ("lasso", "lasso_book")):
            m = p[p["method"] == method]
            if not len(m):
                continue
            r = m.iloc[0]
            n = float(r["n_blocks"])
            row[f"{pre}_mean"] = float(r["blockmean"])
            row[f"{pre}_se"] = float(r["blockstd"]) / np.sqrt(n)
            row[f"{pre}_hit"] = float(r["hit"])
            row[f"{pre}_blocks"] = n
            row[f"{pre}_n_avail"] = float(r["mean_n_factors"])
            rec = n_obs.get(f"{label}::{method}")
            if rec:
                row[f"{pre}_n_obs"] = rec["n_obs_recorded_blocks"]
                if n_trials:
                    row[f"{pre}_defl"] = deflated_ic(
                        best_abs_ic=abs(row[f"{pre}_mean"]),
                        n_obs=rec["n_obs_recorded_blocks"],
                        n_trials=int(n_trials))["deflated_ic"]
    jsl = PIT / f"{label}.jsonl"
    if jsl.exists():
        sel = [json.loads(l) for l in jsl.read_text().splitlines()]
        sel = [r["diag"].get("n_nonzero_coef") for r in sel
               if r["method"] == "lasso" and isinstance(r.get("diag"), dict)]
        sel = [x for x in sel if x is not None]
        if sel:
            row["lasso_n_selected"] = float(np.mean(sel))
    return row


def f(x, nd=4, dash="--"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{nd}f}"


def main() -> None:
    rows = []
    for tag, name, arm, label in ARMS:
        r = ftse_row(tag, name, arm, label)
        for k, v in nasdaq_ref(tag).items():
            r[f"ndx_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    keep = ["arm", "name", "run",
            "preq_mean", "preq_se", "preq_defl", "preq_hit", "preq_blocks",
            "lasso_book_mean", "lasso_book_se", "lasso_book_defl",
            "lasso_book_hit", "lasso_book_blocks",
            "med_abs_ic", "flip_share", "flip_n",
            "n_computable", "n_failed", "n_eff", "mean_abs_corr",
            "n_trials", "cost_usd",
            "preq_n_obs", "lasso_book_n_obs",
            "preq_n_avail", "lasso_n_selected",
            "ndx_preq_mean", "ndx_lasso_book_mean", "ndx_med_abs_ic",
            "ndx_flip_share", "ndx_n_book", "ndx_n_eff", "ndx_mean_abs_corr"]
    df = df.reindex(columns=keep)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "transfer_table.csv", index=False)

    lines = [
        r"\begin{tabular}{|c|l|rrrc|rrrc|rr|rrr|rr|}",
        r"\hline",
        r"\multirow{2}{*}{\#} & \multirow{2}{*}{Arm} & "
        r"\multicolumn{4}{c|}{Prequential (LightGBM)$^{\dagger}$} & "
        r"\multicolumn{4}{c|}{Walk-forward (Lasso)} & "
        r"\multicolumn{2}{c|}{Per factor} & "
        r"\multicolumn{3}{c|}{Independence} & "
        r"\multicolumn{2}{c|}{Cost$^{\ddagger}$} \\",
        r" & & $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & "
        r"$\mathrm{IC}_{\mathrm{defl}}$ & $H$ "
        r"& $\overline{\mathrm{IC}}$ & $s/\sqrt{10}$ & "
        r"$\mathrm{IC}_{\mathrm{defl}}$ & $H$ "
        r"& $\widetilde{|\overline{\mathrm{IC}}|}$ & $\Phi$ "
        r"& $|\mathcal B|$ & $N_{\mathrm{eff}}$ & $\bar{|\rho|}$ "
        r"& $N_{\mathrm{tr}}$ & \$ \\",
        r"\hline",
    ]
    notes = []
    for _, r in df.iterrows():
        def mean_cell(pre):
            v = f(r.get(f"{pre}_mean"))
            nb = r.get(f"{pre}_blocks")
            if nb is not None and np.isfinite(nb) and nb < 10:
                mark = chr(ord("a") + len(notes))
                notes.append(
                    f"$^{{{mark}}}$ mean over {int(nb)}/10 defined blocks "
                    "(LassoCV selected zero factors in the remaining "
                    "blocks, so those composites are constant and their "
                    "block IC is undefined)")
                return v + f"$^{{{mark}}}$"
            return v
        cells = [
            str(r["arm"]),
            r"\texttt{" + str(r["name"]).replace("_", r"\_") + "}",
            mean_cell("preq"), f(r.get("preq_se")),
            f(r.get("preq_defl")), f(r.get("preq_hit"), 1),
            mean_cell("lasso_book"), f(r.get("lasso_book_se")),
            f(r.get("lasso_book_defl")), f(r.get("lasso_book_hit"), 1),
            f(r.get("med_abs_ic")), f(r.get("flip_share"), 2),
            f(r.get("n_computable"), 0), f(r.get("n_eff"), 1),
            f(r.get("mean_abs_corr"), 2),
            f(r.get("n_trials"), 0), f(r.get("cost_usd"), 2),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", "",
              r"% $^{\dagger}$ On this panel no search ran: for ALL rows the "
              r"prequential column is a fixed-book PIT LightGBM refit (the "
              r"thesis's 101-alpha convention); the thesis column for arms "
              r"1/4/6 is the live run's own prequential record.",
              r"% $^{\ddagger}$ $N_{\mathrm{tr}}$ and \$ are the ORIGINAL "
              r"Nasdaq-100 search's trial count and metered cost, carried "
              r"over unchanged: the search was paid for once and nothing "
              r"was spent on the FTSE panel.  $\mathrm{IC}_{\mathrm{defl}}$ "
              r"deflates the FTSE record for that trial count at the FTSE "
              r"record's own n_obs."]
    lines += ["% " + n for n in notes]
    (OUT / "transfer_table.tex").write_text("\n".join(lines) + "\n")
    write_report(df)
    print(df.to_string(index=False))
    print("wrote", OUT / "transfer_table.csv", ".tex and REPORT.md")


def block_bar_counts() -> list[tuple[int, str, str, int]]:
    """(generation, start, end, n_bars on the UK calendar) per WF block."""
    store = OUT / "signal_store"
    parqs = sorted(store.glob("*.parquet"))
    if not parqs:
        return []
    idx = pd.read_parquet(parqs[0]).index
    prq = (REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/"
           "L4WF_terra_s0/evolution/prequential.jsonl")
    rows = []
    for line in prq.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            n = int(((idx >= pd.Timestamp(r["start"]))
                     & (idx < pd.Timestamp(r["end"]))).sum())
            rows.append((r["generation"], r["start"][:10], r["end"][:10], n))
    return sorted(rows)


def write_report(df: pd.DataFrame) -> None:
    store = OUT / "signal_store"
    parqs = sorted(store.glob("*.parquet"))
    shape = ""
    if parqs:
        s = pd.read_parquet(parqs[0])
        shape = f"{s.shape[0]} bars x {s.shape[1]} tickers"
    blocks = block_bar_counts()
    ver = json.loads((DER / "verifications.json").read_text()) \
        if (DER / "verifications.json").exists() else {}
    pw_f = json.loads((DER / "diversity_pairwise_ftse.json").read_text()) \
        if (DER / "diversity_pairwise_ftse.json").exists() else {}
    pw_n = json.loads((DER / "diversity_pairwise_nasdaq.json").read_text()) \
        if (DER / "diversity_pairwise_nasdaq.json").exists() else {}
    lines = [
        "# FTSE 100 transfer evaluation", "",
        "Already-published Nasdaq-100 factor books re-scored, unchanged, on a "
        "FTSE 100 panel (`quant.config.ftse100.yaml`, provider `fmp_archive`, "
        "archive `data/vendor/fmp_uk`, 2010-01-01 to 2026-07-27, "
        f"{shape}, close density 0.9075 vs 0.463 on the Nasdaq WF panel). "
        "No research, evolution or factor selection ran on this panel; the "
        "books, the fit-window convention (< 2021-07-20), the 10 prequential "
        "blocks and every metric definition are exactly those of the thesis "
        "(`scripts/thesis_ablation_master_table.py`).", "",
        "## Caveats", "",
        "* **Prequential column is not strictly like-for-like with the "
        "thesis**: on this panel no search ran, so for ALL FOUR rows the "
        "'Prequential (LightGBM)' record is a fixed-book PIT LightGBM refit "
        "- the convention the thesis applies to its 101-alpha row - whereas "
        "the thesis column for arms 1/4/6 is the live run's own prequential "
        "record (the archive was still evolving while those blocks were "
        "traded).",
        "* **N_tr and $ are the Nasdaq search's numbers, carried over "
        "unchanged**: they are properties of the one-time search (arm 4 = "
        "the L1H_terra_s0b rerun, 86 trials / $14.79 from its own "
        "state.json + llm_usage.json; master_table.csv row 4 holds the older "
        "L1H_terra_s0 and is not used). Nothing was spent on the FTSE panel "
        "and no trial was run here; IC_defl deflates the FTSE record for "
        "the ORIGINAL trial count at the FTSE record's own n_obs.",
        "* **n_obs**: recovered from the saved per-block composite "
        "predictions (sum of `_pooled_ic` valid-pair counts over the "
        "record's defined blocks): "
        + "; ".join(
            f"{r['run']} GBM {int(r['preq_n_obs'])} / Lasso "
            f"{int(r['lasso_book_n_obs'])}"
            for _, r in df.iterrows()
            if np.isfinite(r.get("preq_n_obs") or np.nan)) +
        ". The thesis's Nasdaq records pool 127,911 observations; the "
        "full-record FTSE counts are ~118k (100 tickers, denser panel), "
        "so the luck term per trial is slightly larger here (e.g. "
        "0.0087 at N_tr=86, 0.0107-0.0113 at N_tr=890).",
        "* **Survivorship**: FMP serves no historical FTSE constituent table "
        "on this subscription, so the universe is the CURRENT FTSE 100 "
        "membership held static over 2010-2026 - survivorship-biased by "
        "construction (accepted: the experiment measures correlations, not a "
        "tradable strategy). The high close density (0.91 vs 0.46) reflects "
        "the absence of a per-bar membership mask; the missing cells are "
        "names that listed after 2010.",
        "* **filingDate**: outside the US, FMP's `filingDate` equals the "
        "fiscal period end (median lag 0 days on the UK names tested). The "
        "`_filed_on` guard in `fmp_archive.py` rejects `filed <= period_end` "
        "and falls back to `period_end + reporting_lag_days` (60), so there "
        "is no look-ahead - but the fundamental availability stamp is a "
        "60-day convention, not an observed filing date.",
        "* **Reporting cadence**: UK non-financials report semi-annually "
        "(e.g. VOD.L, ULVR.L show 6-month gaps between 'quarterly' "
        "statements); banks and oil majors report quarterly. Quarterly-"
        "stepped fundamentals are therefore often half-year-stepped here.",
        "* **Block bar counts**: the WF blocks are defined by the runs' "
        "calendar dates; on the UK trading calendar a block holds 127-142 "
        "index bars instead of exactly 126 (see below) - expected and "
        "correct. Counts above ~131 are inflated by a vendor quirk: FMP "
        "returns 102 Sunday-stamped price rows (2019-11 to 2025-03) for "
        "exactly one dual-listed name, CCEP.L - 0.026% of the panel's "
        "finite close cells. Every other name is NaN on those rows, so "
        "they only enter the IC pools through CCEP itself; left in place "
        "rather than cleaned, since removing them would be a panel edit "
        "this transfer run deliberately avoids.",
        "* **Short Lasso records**: LDU8's Lasso mean rests on 6/10 blocks "
        "and L4WF's on 9/10 - in the missing blocks LassoCV selected ZERO "
        "factors (n_nonzero_coef=0, no fit error), the composite is "
        "constant and its block IC undefined. Marked in the table itself.",
        "",
        "## Per-block bar counts (UK calendar)", "",
        "| gen | start | end | bars |", "|---|---|---|---|",
    ]
    for g, s0, e0, n in blocks:
        lines.append(f"| {g} | {s0} | {e0} | {n} |")

    # late-lister handling verification
    if ver:
        a, b = ver.get("ic_path", {}), ver.get("fit_path", {})
        lines += [
            "", "## Late-listed names: empirical verification", "",
            "Fifteen of the 100 names listed after 2010; their pre-listing "
            "cells are NaN, exactly like a non-member cell on the Nasdaq "
            "panel. Verified on this panel:", "",
            f"* **IC path** (block {a.get('block')}): "
            f"{a.get('n_assets_entering_ic')}/{a.get('n_assets_total')} "
            "assets entered the pooled IC; per-asset valid-pair counts: "
            + ", ".join(f"{k} {v}" for k, v in
                        (a.get("per_asset_pairs_probe") or {}).items()) +
            ". Late listers (HLN.L listed 2022, MTLN.L 2025) contribute 0 "
            "pairs in 2021 blocks and are skipped; the pooled mean is "
            "weighted by each asset's valid-pair count "
            "(`_weighted_asset_pearson`), so they cannot corrupt it.",
            f"* **Combiner fit** (block {b.get('block')}): of "
            f"{b.get('candidate_rows')} candidate (bar, name) training rows, "
            f"{b.get('train_rows')} have a finite forward return and were "
            f"kept - {100 * (b.get('dropped_share') or 0):.1f}% dropped, "
            "matching the pre-listing NaN share. Within surviving rows a "
            "missing signal cell becomes z=0 (its own fit-window mean) - "
            "neutral, identical to the Nasdaq treatment.",
        ]

    # three-way diversity decomposition
    zc_f = json.loads((DER / "diversity_zeroed_std_ftse.json").read_text()) \
        if (DER / "diversity_zeroed_std_ftse.json").exists() else {}
    zc_n = json.loads((DER / "diversity_zeroed_std_nasdaq.json").read_text()) \
        if (DER / "diversity_zeroed_std_nasdaq.json").exists() else {}
    if pw_f and pw_n and zc_f and zc_n:
        lines += [
            "", "## Diversity: three-way decomposition (both panels)", "",
            "The thesis diversity statistic - definition **(a)** "
            "(`wf_arm_factor_analysis.py`) - stacks RAW, un-standardised "
            "signals and `np.nan_to_num`s missing cells to 0 before "
            "`np.corrcoef` over all pooled cells. Definition **(b)** "
            "changes two things at once: signals are z-scored per "
            "underlying over the fit window, AND each pairwise "
            "correlation uses only cells where both signals and the "
            "underlying's close are finite (min overlap 1000 cells). "
            "Definition **(c)** separates them: z-scored exactly as in "
            "(b), but with (a)'s missing-cell handling (NaN -> 0, all "
            "pooled cells). So (a)->(c) isolates the standardisation "
            "change and (c)->(b) isolates the missing-cell handling.",
            "",
            "| book | panel | (a) mean abs rho / N_eff | (c) z-scored, zeros | (b) pairwise-complete |",
            "|---|---|---|---|---|",
        ]
        nas_thesis = {"LDU8_terra_s0": (0.172, 9.5), "L1H_terra_s0b": (0.10, 12.2),
                      "L4WF_terra_s0": (0.088, 22.0), "zoo": (0.262, 7.6)}

        def cell(d):
            return (f"{d['mean_abs_corr']:.3f} / "
                    f"{d['effective_n_participation_ratio']:.1f}")

        for arm in ("LDU8_terra_s0", "L1H_terra_s0b", "L4WF_terra_s0", "zoo"):
            ft = json.loads((ANA / arm / "diversity.json").read_text())
            nt = nas_thesis[arm]
            lines.append(f"| {arm} | FTSE | {cell(ft)} | {cell(zc_f[arm])} | "
                         f"{cell(pw_f[arm])} |")
            lines.append(f"| {arm} | Nasdaq | {nt[0]:.3f} / {nt[1]:.1f} | "
                         f"{cell(zc_n[arm])} | {cell(pw_n[arm])} |")
        lines += [
            "",
            "**Attribution**: the standardisation step (a)->(c) carries "
            "virtually the entire move, for the zoo AND for the "
            "language-model books alike - on Nasdaq the zoo goes 0.262 / "
            "7.6 -> 0.104 / 30.2 at (c), and the further step to "
            "pairwise-complete correlation (c)->(b) then changes "
            "mean|rho| by at most 0.010 and N_eff by at most 3.3 (both "
            "extremes the FTSE zoo; every other book moves <= 0.9 on "
            "N_eff). The mechanism: once signals are z-scored per "
            "underlying, an imputed 0 IS the fit-window mean and is "
            "nearly neutral in a correlation - the shared-missingness "
            "artefact in (a) is not the zeros per se but the zeros "
            "interacting with un-centred, un-scaled raw signals (the "
            "zoo's raw alphas are the least centred, so it is hit "
            "hardest). Consequences: (a)'s absolute N_eff levels and any "
            "cross-panel comparison are unreliable, while within-panel "
            "the ordering by mean|rho| (zoo most redundant per member) "
            "survives under all three definitions. The below-overlap "
            "pairs in (b) for L1H (21 pairs, both panels) all involve "
            "`sector_synchronous_volshock_muted_receiver`: its signal is "
            "CONSTANT per underlying over the fit window (100/100 FTSE "
            "columns), so its z-score is undefined - the same factor "
            "already documented on Nasdaq as yielding no per-factor IC "
            "in any prequential block; in (c) it enters as an all-zero "
            "column.",
            "",
            "**Effective-factor share N_eff/|B| (Nasdaq panel)** - the "
            "thesis quotes arms 4/6 at 'five to seven times' the zoo's "
            "share under (a); under the corrected definitions that "
            "factor shrinks to ~2-3x:",
            "",
            "| definition | arm 4 L1H | arm 6 L4WF | zoo | L1H/zoo | L4WF/zoo |",
            "|---|---|---|---|---|---|",
            "| (a) thesis | 0.555 | 0.386 | 0.075 | 7.4x | 5.1x |",
            "| (c) z-scored, zeros | 0.811 | 0.598 | 0.299 | 2.7x | 2.0x |",
            "| (b) pairwise-complete | 0.849 | 0.605 | 0.281 | 3.0x | 2.2x |",
        ]

    lines += ["", "## Failed / non-computable factors", ""]
    for _, r in df.iterrows():
        arm = r["run"]
        div = ANA / arm / "diversity.json"
        if div.exists():
            j = json.loads(div.read_text())
            failed = j.get("failed") or []
            lines.append(f"* **{arm}**: {j.get('n_failed', 0)} failed in the "
                         "per-factor analysis")
            for fr in failed:
                lines.append(f"    * `{fr['factor_id']}`: {fr['error']}")
    lines += [
        "* **L4WF snapshots race only**: "
        "`concordant_revision_underreaction_drift` (a kept_pool member, NOT "
        "in the published 57-factor book) fails with 'Can only compare "
        "identically-labeled DataFrames'. Reason: the factor compares "
        "`np.sign(eps_revision) == np.sign(revenue_revision)` as "
        "DataFrames; on the FTSE panel the loader serves `epsEstimate` "
        "with 94 columns and `revenueEstimate` with 95 (FMP has no "
        "analyst estimates for the investment trusts ALW.L / FCIT.L / "
        "PSH.L / PCT.L / SMT.L, and additionally none for MTLN.L's eps), "
        "so the two frames' column sets differ by one name and pandas "
        "refuses the comparison; on the Nasdaq panel both fields cover "
        "the identical column set. The race ran without it (210/211 "
        "signals).",
    ]
    lines += [
        "", "## Reading", "",
        "* The 101 formulaic alphas transfer almost fully: PIT Lasso 0.0654 "
        "(10/10 positive blocks) vs 0.0670 on Nasdaq, and their median "
        "per-factor |block IC| is slightly HIGHER on FTSE (0.0244 vs "
        "0.0207). All 18 IndNeutralize alphas computed (UK sector/industry "
        "labels present, `sector` density 1.00).",
        "* The evolved book (arm 6, L4WF) transfers at reduced strength: "
        "PIT GBM 0.0352 (vs its Nasdaq prequential 0.0352 - numerically "
        "similar by coincidence, the FTSE number is a fixed-book refit, "
        "not a live run), Lasso 0.0404 over 9 defined blocks (vs 0.0717); "
        "the ic-weighted combiner reaches 0.0516 at 10/10. Per-factor "
        "median |IC| 0.0123 matches its Nasdaq value (0.0114) and the "
        "flip share is unchanged (0.25 vs 0.21) - the individual factors "
        "carry over, the combined linear record gives up roughly half.",
        "* The graph-grounded seeding book (arm 4, L1H_terra_s0b) also "
        "transfers: Lasso 0.0382 at 10/10 (vs 0.0653), equal/ic/ridge "
        "0.044-0.052 all at 10/10; only the GBM combiner is weak here "
        "(0.011, hit 0.4).",
        "* The ungrounded book (arm 1, LDU8) does NOT transfer: every "
        "combiner is at or below zero (best equal-weight 0.0044; Lasso "
        "-0.0069 on the 6 defined blocks), and its flip share rises to "
        "0.48. Its Nasdaq record was already the weakest (preq 0.0070, "
        "Lasso 0.0138).",
        "* Deflation (FTSE n_obs, original N_tr): L4WF stays clearly "
        "above its luck term (GBM 0.0352 -> 0.0244, Lasso 0.0404 -> "
        "0.0291 at N_tr=890) and so does L1H's Lasso record (0.0382 -> "
        "0.0296 at N_tr=86), while L1H's GBM record barely survives "
        "(0.0110 -> 0.0023) and LDU8's Lasso record deflates to 0. "
        "CAUTION on LDU8's GBM cell: that record's MEAN IS NEGATIVE "
        "(-0.0173); the thesis definition deflates the magnitude, so "
        "its 0.0086 says only that the wrong-signed record exceeds the "
        "luck term - it is not evidence of an edge. The zoo carries no "
        "trial count, so its IC_defl is '--' exactly as in the thesis.",
        "* Ordering is preserved: zoo > L4WF ~ L1H >> LDU8 on the combined "
        "records, matching the Nasdaq ladder's reading that grounding "
        "carries the value and that the ungrounded arm's record does not "
        "generalise.",
        "", "## Table", "", df.to_string(index=False), ""]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
