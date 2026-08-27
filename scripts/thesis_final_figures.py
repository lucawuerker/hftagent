"""Final results-chapter figures for the thesis.

Arm -> run mapping (user ground truth, 2026-08-18):
    1  LDU8_terra_s0     ungrounded, no evolution
    2  LDP8_terra_s0     24 random papers, no graph
    3  LDG_terra_s0b     graph briefs, no papers   (replication rerun)
    4  L1H_terra_s0b     graph + papers            (replication rerun)
    5  L2WFP_terra_s0    ungrounded + evolution
    6  L4WF_terra_s0     grounded + evolution
    7  L5WF_terra_s0     grounded + evolution + debate
    8  L1HBD_terra_s0    grounded seeding + debate (no evolution)

Everything is read from existing analysis outputs; no expensive recomputation.
Arms 1,2,5,6,7,8 come from thesis_ablation/tables/{ladder_summary,master_table}.csv;
arms 3,4 are recomputed from wf_arm_analysis_local/<run>/ + pit_combiners/ with the
exact ladder definitions (preq = mean combined OOS IC over generations 11-20;
lasso book/pool = mean of 10 PIT block ICs; med |IC| and flip share on the
finite-and-nonzero ic_fit subset; N_eff = participation ratio).

Output: data/comparisons/thesis_final_figures/*.png (300 dpi, no titles)
plus arm_metrics.csv / contrast_deltas.csv for cross-checking.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "data/comparisons/thesis_ablation/tables"
ANA = REPO / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
PRERUNS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
OUT = REPO / "data/comparisons/thesis_final_figures"

ARM_RUNS = {
    "1": "LDU8_terra_s0",
    "2": "LDP8_terra_s0",
    "3": "LDG_terra_s0b",
    "4": "L1H_terra_s0b",
    "5": "L2WFP_terra_s0",
    "6": "L4WF_terra_s0",
    "7": "L5WF_terra_s0",
    "8": "L1HBD_terra_s0",
}
ARM_NAMES = {
    "1": "LDU8", "2": "LDP8", "3": "LDG", "4": "L1H",
    "5": "L2WFP", "6": "L4WF", "7": "L5WF", "8": "L1HBD",
}
# ladder_summary rows are keyed by the ORIGINAL run ids / arm tags
LADDER_ARM_FOR = {"1": "1", "2": "2", "5": "5", "6": "6", "7": "7", "8": "8b"}

# dataviz reference palette
BLUE = "#2a78d6"       # improvement
RED = "#c94040"        # deterioration
INK = "#1a1a18"
INK_MUTED = "#6e6d68"
GRID = "#e4e2dd"
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#2a78d6"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------- data assembly
def _pit_row(label: str, method: str = "lasso") -> tuple[float, float]:
    p = pd.read_csv(PIT / f"{label}_summary.csv")
    r = p[p["method"] == method].iloc[0]
    n = float(r["n_blocks"])
    return float(r["blockmean"]), float(r["blockstd"]) / math.sqrt(n)


def _s0b_row(run: str, cur_label: str) -> dict:
    d = ANA / run
    preq = pd.read_csv(d / "prequential_record.csv")
    ics = preq[preq["generation"] >= 11]["combined_oos_ic"].dropna().to_numpy()
    pf = pd.read_csv(d / "per_factor_blocks.csv")
    ok = pf.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    div = json.loads((d / "diversity.json").read_text())
    usage = json.loads((PRERUNS / run / "evolution/llm_usage.json").read_text())
    cost = sum(v.get("cost_usd", 0.0) for v in usage.get("by_role", {}).values())
    book_m, book_se = _pit_row(cur_label)
    pool_m, pool_se = _pit_row(run)
    return {
        "run": run,
        "preq_mean": float(ics.mean()),
        "preq_se": float(ics.std(ddof=1)) / math.sqrt(len(ics)),
        "lasso_book_mean": book_m, "lasso_book_se": book_se,
        "lasso_pool_mean": pool_m, "lasso_pool_se": pool_se,
        "med_abs_ic": float(ok["ic_wf_blockmean"].abs().median()),
        "flip_share": float(
            (np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()),
        "n_eff": float(div["effective_n_participation_ratio"]),
        "mean_abs_corr": float(div["mean_abs_corr"]),
        "n_book_analysed": int(div["n_factors"]),
        "cost_usd": cost,
    }


def assemble() -> pd.DataFrame:
    lad = pd.read_csv(TAB / "ladder_summary.csv", dtype={"arm": str})
    mas = pd.read_csv(TAB / "master_table.csv", dtype={"arm": str})
    rows = []
    for arm, run in ARM_RUNS.items():
        if arm in ("3", "4"):
            cur = {"3": "LDGCUR_terra_s0b", "4": "L1HCUR_terra_s0b"}[arm]
            row = _s0b_row(run, cur)
        else:
            lr = lad[lad["arm"] == LADDER_ARM_FOR[arm]].iloc[0]
            mr = mas[mas["arm"] == LADDER_ARM_FOR[arm]].iloc[0]
            row = {
                "run": run,
                "preq_mean": lr["preq_mean"], "preq_se": lr["preq_se"],
                "lasso_book_mean": lr["lasso_book_mean"],
                "lasso_book_se": lr["lasso_book_se"],
                "lasso_pool_mean": lr.get("lasso_pool_mean"),
                "lasso_pool_se": lr.get("lasso_pool_se"),
                "med_abs_ic": mr["med_abs_ic"],
                "flip_share": lr["flip_share"],
                "n_eff": lr["n_eff"],
                "mean_abs_corr": lr["mean_abs_corr"],
                "n_book_analysed": lr["n_book_analysed"],
                "cost_usd": lr["cost_usd"],
            }
        row["arm"] = arm
        row["name"] = ARM_NAMES[arm]
        row["eff_share"] = row["n_eff"] / row["n_book_analysed"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("arm")
    return df


# ---------------------------------------------------------------- contrasts
# (main label, qualifier, arm OFF, arm ON); delta = ON - OFF
CONTRASTS = [
    ("evolution", "ungrounded", "1", "5"),
    ("evolution", "grounded", "4", "6"),
    ("retrieval", "with evolution", "5", "6"),
    ("graph briefs", "with papers", "2", "4"),
    ("papers", "with graph briefs", "3", "4"),
    ("debate", "with evolution", "6", "7"),
    ("debate", "seeding only", "4", "8"),
]
GROUP_BREAKS = {3, 5}  # separator above contrast index 3 and 5

# metric key -> (x-axis label, se key or None, scale, lower_is_better)
METRICS = {
    "preq_mean": ("$\\Delta$ prequential mean IC", "preq_se", 1.0, False),
    "lasso_book_mean": ("$\\Delta$ walk-forward Lasso IC (curated book)",
                        "lasso_book_se", 1.0, False),
    "lasso_pool_mean": ("$\\Delta$ walk-forward Lasso IC (kept pool)",
                        "lasso_pool_se", 1.0, False),
    "med_abs_ic": ("$\\Delta$ median per-factor $|$IC$|$", None, 1.0, False),
    "flip_share": ("$\\Delta$ sign-flip share (pp)", None, 100.0, True),
    "eff_share": ("$\\Delta$ effective-factor share (pp)", None, 100.0, False),
    "mean_abs_corr": ("$\\Delta$ mean $|\\rho|$", None, 1.0, True),
}


def delta_chart(df: pd.DataFrame, metric: str, fname: str) -> None:
    xlabel, se_key, scale, lower_better = METRICS[metric]
    rows = []
    for label, qual, a, b in CONTRASTS:
        va, vb = df.loc[a, metric], df.loc[b, metric]
        if pd.isna(va) or pd.isna(vb):
            continue
        delta = (vb - va) * scale
        se = np.nan
        if se_key is not None:
            sa, sb = df.loc[a, se_key], df.loc[b, se_key]
            if pd.notna(sa) and pd.notna(sb):
                se = math.sqrt(sa ** 2 + sb ** 2) * scale
        rows.append((label, qual, a, b, delta, se))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * n + 1.2))
    ys = np.arange(n)[::-1]
    kept_idx = [i for i, c in enumerate(CONTRASTS)
                if not (pd.isna(df.loc[c[2], metric]) or pd.isna(df.loc[c[3], metric]))]
    for y, (label, qual, a, b, delta, se) in zip(ys, rows):
        improves = (delta < 0) if lower_better else (delta > 0)
        ax.barh(y, delta, height=0.55, color=BLUE if improves else RED,
                zorder=3)
        if pd.notna(se):
            ax.errorbar(delta, y, xerr=se, fmt="none", ecolor=INK_MUTED,
                        elinewidth=1.1, capsize=3, zorder=4)
        ax.text(-0.015, y + 0.12, label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=11)
        ax.text(-0.015, y - 0.24, f"{qual} · {a}→{b}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, color=INK_MUTED)
    # group separators (only between rows that are actually present)
    for brk in GROUP_BREAKS:
        present_above = [i for i in kept_idx if i < brk]
        present_below = [i for i in kept_idx if i >= brk]
        if present_above and present_below:
            k = len(present_above)
            ax.axhline(ys[k - 1] - 0.5, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=1.0, zorder=2)
    ax.set_yticks([])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.28)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2x2 grids
def _fmt(metric: str, val: float, se: float | None) -> str:
    if metric in ("flip_share", "eff_share"):
        return f"{100 * val:.0f}%"
    if metric == "mean_abs_corr":
        return f"{val:.3f}"
    s = f"{val:.4f}" if metric == "preq_mean" else f"{val:.3f}"
    if se is not None and pd.notna(se):
        s += f" $\\pm$ {se:.4f}" if metric == "preq_mean" else f" $\\pm$ {se:.3f}"
    return s


def grid_2x2(df: pd.DataFrame, metric: str, fname: str) -> None:
    _, se_key, _, lower_better = METRICS[metric]
    # (row, col): rows = graph off/on (top=off), cols = papers off/on
    cells = {(0, 0): "1", (0, 1): "2", (1, 0): "3", (1, 1): "4"}
    vals = {k: df.loc[a, metric] for k, a in cells.items()}
    lo, hi = min(vals.values()), max(vals.values())
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for (r, c), arm in cells.items():
        v = vals[(r, c)]
        t = 0.5 if hi == lo else (v - lo) / (hi - lo)
        if lower_better:
            t = 1.0 - t
        color = SEQ_RAMP[int(round(t * (len(SEQ_RAMP) - 3)))]  # cap at mid-dark
        ax.add_patch(plt.Rectangle((c + 0.02, 1 - r + 0.02), 0.96, 0.96,
                                   facecolor=color, edgecolor="white", lw=2))
        se = df.loc[arm, se_key] if se_key else None
        ax.text(c + 0.5, 1 - r + 0.60, f"{arm}  {ARM_NAMES[arm]}",
                ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(c + 0.5, 1 - r + 0.38, _fmt(metric, v, se),
                ha="center", va="center", fontsize=12)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["no papers", "papers"], fontsize=12)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["graph\nbriefs", "no graph"], fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("auto")
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- cost panel
def cost_chart(df: pd.DataFrame, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    arms = list(ARM_RUNS)
    costs = [df.loc[a, "cost_usd"] for a in arms]
    xs = np.arange(len(arms))
    ax.bar(xs, costs, width=0.62, color=BLUE, zorder=3)
    for x, cst in zip(xs, costs):
        ax.text(x, cst + max(costs) * 0.015, f"{cst:.0f}",
                ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{a}\n{ARM_NAMES[a]}" for a in arms], fontsize=9.5)
    ax.set_ylabel("LLM cost (USD)")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = assemble()
    df.to_csv(OUT / "arm_metrics.csv")

    deltas = []
    for label, qual, a, b in CONTRASTS:
        rec = {"contrast": f"{label} ({qual})", "arms": f"{a}->{b}"}
        for m in METRICS:
            va, vb = df.loc[a, m], df.loc[b, m]
            rec[f"delta_{m}"] = (vb - va) if pd.notna(va) and pd.notna(vb) else np.nan
        deltas.append(rec)
    pd.DataFrame(deltas).to_csv(OUT / "contrast_deltas.csv", index=False)

    for metric, tag in [
        ("preq_mean", "preq"),
        ("lasso_book_mean", "lasso_book"),
        ("lasso_pool_mean", "lasso_pool"),
        ("med_abs_ic", "med_abs_ic"),
        ("flip_share", "flip_share"),
        ("eff_share", "eff_share"),
        ("mean_abs_corr", "mean_abs_corr"),
    ]:
        delta_chart(df, metric, f"delta_{tag}.png")
        grid_2x2(df, metric, f"grid2x2_{tag}.png")
    cost_chart(df, "cost_per_arm.png")

    print(df[["name", "preq_mean", "preq_se", "lasso_book_mean",
              "lasso_pool_mean", "med_abs_ic", "flip_share", "eff_share",
              "mean_abs_corr", "cost_usd"]].to_string())
    print(f"\nwrote {len(list(OUT.glob('*.png')))} PNGs -> {OUT}")


if __name__ == "__main__":
    main()
