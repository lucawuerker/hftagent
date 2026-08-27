"""Thesis ablation-chapter figures (one PNG per figure) from the tables built
by scripts/thesis_ablation_derive.py.

Outputs to data/comparisons/thesis_ablation/figures/.
Re-runnable; figures with missing inputs are skipped with a note.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

REPO = Path(__file__).resolve().parents[1]
TAB = REPO / "data/comparisons/thesis_ablation/tables"
FIG = REPO / "data/comparisons/thesis_ablation/figures"

# validated categorical palette (light mode), assigned by component family
C_IDEA = "#2a78d6"      # ideation-only arms
C_EVO = "#eb6834"       # + evolution
C_REV = "#1baf7a"       # + adversarial review
C_GP = "#6b6a67"        # non-LLM GP baseline
C_MQ = "#e87ba4"        # weak-model arm
C_GRID = "#e4e3df"
C_TEXT = "#0b0b0b"
C_MUT = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "thesis_div", ["#2a78d6", "#f2f1ee", "#e34948"])

ORDER = ["1", "2", "3", "4", "5", "6", "7", "8a", "8b", "9"]


def family_color(row) -> str:
    if row["llm"].startswith("none"):
        return C_GP
    if row["llm"] == "GPT-4o-mini":
        return C_MQ
    if row["review"]:
        return C_REV
    if row["evolution"]:
        return C_EVO
    return C_IDEA


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUT, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load():
    ladder = pd.read_csv(TAB / "ladder_summary.csv", dtype={"arm": str})
    ladder["arm"] = pd.Categorical(ladder["arm"], ORDER + ["MQ"],
                                   ordered=True)
    ladder = ladder.sort_values("arm").reset_index(drop=True)
    preq = pd.read_csv(TAB / "prequential_blocks.csv")
    return ladder, preq


def f01_ladder(ladder):
    d = ladder[ladder["arm"] != "MQ"].dropna(subset=["preq_mean"])
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    cols = [family_color(r) for _, r in d.iterrows()]
    x = np.arange(len(d))
    ax.bar(x, d["preq_mean"], yerr=d["preq_se"], width=0.62, color=cols,
           error_kw=dict(ecolor=C_MUT, lw=1.1, capsize=3))
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, r["preq_mean"] + r["preq_se"] + 0.004,
                f"{r['preq_mean']:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=C_TEXT)
    ax.axhline(0, color=C_MUT, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{n}" for a, n in zip(d["arm"], d["name"])],
                       fontsize=8.5)
    ax.set_ylabel("prequential IC, mean over 10 WF blocks", fontsize=10)
    style_ax(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (C_IDEA, C_EVO, C_REV, C_GP)]
    ax.legend(handles, ["ideation only", "+ evolution",
                        "+ adversarial review", "GP baseline (no LLM)"],
              frameon=False, fontsize=8.5, ncol=4, loc="upper left",
              bbox_to_anchor=(0, 1.12))
    save(fig, "f01_ladder_prequential.png")


def f02_block_heatmap(ladder, preq):
    d = ladder[ladder["arm"] != "MQ"]
    runs = d["run"].tolist()
    names = [f"{a}  {n}" for a, n in zip(d["arm"], d["name"])]
    mat = np.full((len(runs), 10), np.nan)
    for i, run in enumerate(runs):
        p = preq[preq["run"] == run]
        for _, r in p.iterrows():
            mat[i, int(r["block"]) - 1] = r["ic"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap=DIVERGING, norm=TwoSlopeNorm(0., -vmax, vmax),
                   aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(10):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7.2, color=C_TEXT)
    ax.set_xticks(range(10))
    p0 = preq[preq["run"] == runs[0]].sort_values("block")
    labels = [f"{b}\n{s[:7]}" for b, s in zip(p0["block"], p0["start"])]
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("walk-forward block (first date)", fontsize=9.5)
    ax.tick_params(colors=C_MUT)
    fig.colorbar(im, ax=ax, shrink=0.8, label="prequential IC")
    save(fig, "f02_block_heatmap.png")


def f03_preq_vs_lasso(ladder):
    d = ladder[ladder["arm"] != "MQ"].dropna(
        subset=["preq_mean", "lasso_book_mean"])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r["preq_mean"], r["lasso_book_mean"]], [yi, yi],
                color=C_GRID, lw=2, zorder=1)
    ax.scatter(d["preq_mean"], y, s=52, color=C_EVO, zorder=2,
               label="own prequential (GBM refit)")
    ax.scatter(d["lasso_book_mean"], y, s=52, color=C_IDEA, zorder=2,
               label="PIT Lasso on the published book")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a}  {n}" for a, n in zip(d["arm"], d["name"])],
                       fontsize=9)
    ax.set_xlabel("mean WF block IC", fontsize=10)
    ax.axvline(0, color=C_MUT, lw=0.8)
    style_ax(ax)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, "f03_preq_vs_lasso.png")


def f04_pool_vs_book(ladder):
    d = ladder.dropna(subset=["lasso_pool_mean"])
    d = d[d["arm"].isin(["1", "2", "3", "4", "8a", "8b"])]
    if not len(d):
        return print("skip f04 (no pool races yet)")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(d))
    w = 0.36
    ax.bar(x - w / 2, d["lasso_book_mean"], w, color=C_IDEA,
           label="curated book")
    ax.bar(x + w / 2, d["lasso_pool_mean"], w, color=C_EVO,
           label="full kept pool")
    for xi, (_, r) in zip(x, d.iterrows()):
        if np.isfinite(r["lasso_book_mean"]):
            ax.text(xi - w / 2, r["lasso_book_mean"] + 0.002,
                    f"{r['lasso_book_mean']:.3f}\nn={r['lasso_book_n_avail']:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
        if np.isfinite(r["lasso_pool_mean"]):
            ax.text(xi + w / 2, r["lasso_pool_mean"] + 0.002,
                    f"{r['lasso_pool_mean']:.3f}\nn={r['lasso_pool_n_avail']:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{n}" for a, n in zip(d["arm"], d["name"])],
                       fontsize=8.5)
    ax.set_ylabel("PIT Lasso mean WF block IC", fontsize=10)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "f04_lasso_pool_vs_book.png")


def f05_grounding_2x2(ladder):
    cells = {(0, 0): "1", (0, 1): "2", (1, 0): "3", (1, 1): "4"}
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    for (i, j), arm in cells.items():
        r = ladder[ladder["arm"] == arm].iloc[0]
        y, x = 1 - i, j
        val = r["preq_mean"]
        shade = plt.matplotlib.colors.to_rgba(C_IDEA,
                                              0.12 + 0.75 * min(max(
                                                  val, 0) / 0.06, 1.0))
        ax.add_patch(plt.Rectangle((x + .02, y + .02), .96, .96,
                                   facecolor=shade, edgecolor=C_GRID))
        pool = (f"pool Lasso {r['lasso_pool_mean']:.3f}"
                if np.isfinite(r.get("lasso_pool_mean", np.nan)) else "")
        ax.text(x + .5, y + .58,
                f"{r['arm']}  {r['name']}", ha="center", fontsize=11,
                color=C_TEXT, weight="bold")
        ax.text(x + .5, y + .40,
                f"preq {val:.4f} ± {r['preq_se']:.4f}\nhit {r['preq_hit']:.0%}"
                + (f"\n{pool}" if pool else ""),
                ha="center", va="center", fontsize=9, color=C_TEXT)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["no papers", "papers"], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["graph\nbriefs", "no graph"], fontsize=10)
    ax.tick_params(length=0, colors=C_TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Ideation grounding, prequential record (arms 1–4)",
                 fontsize=11)
    save(fig, "f05_grounding_2x2.png")


def f06_flip_neff(ladder):
    d = ladder[ladder["arm"] != "MQ"].dropna(subset=["flip_share"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(d))
    cols = [family_color(r) for _, r in d.iterrows()]
    axes[0].bar(x, d["flip_share"], width=0.6, color=cols)
    for xi, v in zip(x, d["flip_share"]):
        axes[0].text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8,
                     color=C_TEXT)
    axes[0].set_ylabel("sign-flip share $\\Phi$ (book members)", fontsize=10)
    d2 = ladder[ladder["arm"] != "MQ"].dropna(subset=["n_eff"])
    x2 = np.arange(len(d2))
    cols2 = [family_color(r) for _, r in d2.iterrows()]
    axes[1].bar(x2, d2["n_eff"], width=0.6, color=cols2)
    for xi, (v, n) in zip(x2, zip(d2["n_eff"], d2["n_book_analysed"])):
        axes[1].text(xi, v + 0.3, f"{v:.1f}/{n:.0f}", ha="center",
                     fontsize=8, color=C_TEXT)
    axes[1].set_ylabel("effective N (participation ratio)", fontsize=10)
    for ax, dd in ((axes[0], d), (axes[1], d2)):
        ax.set_xticks(np.arange(len(dd)))
        ax.set_xticklabels(dd["name"], fontsize=8, rotation=45, ha="right")
        style_ax(ax)
    save(fig, "f06_flip_share_and_neff.png")


def f07_cost(ladder):
    d = ladder[ladder["arm"] != "MQ"].dropna(subset=["cost_usd", "preq_mean"])
    d = d[d["cost_usd"] > 0]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for _, r in d.iterrows():
        ax.scatter(r["cost_usd"], r["preq_mean"], s=30 + r["n_trials"] * 0.18,
                   color=family_color(r), alpha=0.85, edgecolor="white",
                   linewidth=1.2, zorder=3)
        ax.annotate(f"{r['arm']} {r['name']}",
                    (r["cost_usd"], r["preq_mean"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=8,
                    color=C_TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("metered LLM cost of the run (USD, log scale)", fontsize=10)
    ax.set_ylabel("prequential mean IC", fontsize=10)
    style_ax(ax)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8)
    ax.text(0.02, 0.02, "marker area $\\propto$ $N_{trials}$",
            transform=ax.transAxes, fontsize=8, color=C_MUT)
    save(fig, "f07_cost_vs_record.png")


def f08_component_deltas(ladder):
    """Matched pairs: what each component adds (prequential mean ± SE)."""
    pairs = [
        ("evolution | ungrounded", "1", "5"),
        ("evolution | grounded", "4", "6"),
        ("retrieval | with evolution", "5", "6"),
        ("review | with evolution", "6", "7"),
        ("review | ideation only", "8a", "8b"),
        ("papers | no graph", "1", "2"),
        ("graph | no papers", "1", "3"),
        ("graph+papers", "1", "4"),
    ]
    rows = []
    for label, a, b in pairs:
        ra = ladder[ladder["arm"] == a].iloc[0]
        rb = ladder[ladder["arm"] == b].iloc[0]
        if not (np.isfinite(ra["preq_mean"]) and np.isfinite(rb["preq_mean"])):
            continue
        rows.append((label, rb["preq_mean"] - ra["preq_mean"],
                     np.hypot(ra["preq_se"], rb["preq_se"]),
                     f"{a}$\\to${b}"))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    cols = [C_EVO if v >= 0 else "#e34948" for v in vals]
    ax.barh(y, vals, xerr=errs, height=0.55, color=cols,
            error_kw=dict(ecolor=C_MUT, lw=1, capsize=3))
    ax.axvline(0, color=C_MUT, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}  ({r[3]})" for r in rows], fontsize=9)
    ax.set_xlabel("$\\Delta$ prequential mean IC (arm B $-$ arm A)",
                  fontsize=10)
    style_ax(ax)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8)
    ax.yaxis.grid(False)
    save(fig, "f08_component_deltas.png")


def f09_model_quality(ladder, preq):
    d = preq[preq["run"].isin(["L1HB_terra_s0", "L1HB_4omini_s0"])]
    if not len(d):
        return print("skip f09")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for run, col, lab in ((("L1HB_terra_s0"), C_IDEA, "L1HB (GPT-5.6 Terra)"),
                          (("L1HB_4omini_s0"), C_MQ,
                           "L1HB (GPT-4o-mini)")):
        p = d[d["run"] == run].sort_values("block")
        ax.plot(p["block"], p["ic"], marker="o", ms=5, lw=2, color=col,
                label=lab)
    ax.axvline(4.5, color=C_MUT, lw=1, ls="--")
    ax.text(4.55, ax.get_ylim()[1] * 0.96,
            "GPT-4o-mini training cutoff (Oct 2023)", fontsize=8,
            color=C_MUT, va="top")
    ax.axhline(0, color=C_MUT, lw=0.8)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("walk-forward block", fontsize=10)
    ax.set_ylabel("prequential block IC", fontsize=10)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "f09_model_quality_blocks.png")


def f10_kg_campaign():
    p = TAB / "kg_campaign.csv"
    if not p.exists():
        return print("skip f10")
    kg = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    for scope, col, lab in (("run", C_IDEA, "per-run book"),
                            ("cum", C_EVO, "cumulative book")):
        d = kg[(kg["scope"] == scope) & (kg["method"] == "ridge")]
        if not len(d):
            continue
        axes[0].plot(d["run"], d["blockmean"], marker="o", ms=4, lw=1.8,
                     color=col, label=lab)
    axes[0].set_xlabel("campaign run", fontsize=10)
    axes[0].set_ylabel("WF mean block IC (ridge)", fontsize=10)
    axes[0].axhline(0, color=C_MUT, lw=0.8)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    for scope, col, lab in (("run", C_IDEA, "per-run"),
                            ("cum", C_EVO, "cumulative")):
        dd = kg[(kg["scope"] == scope) & (kg["method"] == "ridge")]
        if len(dd):
            axes[1].plot(dd["run"], dd["n_factors"], marker="o", ms=4,
                         lw=1.8, color=col, label=lab)
    axes[1].set_xlabel("campaign run", fontsize=10)
    axes[1].set_ylabel("book size (factors)", fontsize=10)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        style_ax(ax)
    save(fig, "f10_kg_campaign.png")


def f11_per_factor_dist():
    p = TAB / "per_factor_all.csv"
    if not p.exists():
        return print("skip f11")
    pf = pd.read_csv(p, dtype={"arm": str})
    pf = pf[pf["arm"] != "MQ"]
    pf["absic"] = pf["ic_wf_blockmean"].abs()
    order = [a for a in ORDER if a in set(pf["arm"])]
    data = [pf[pf["arm"] == a]["absic"].dropna() for a in order]
    names = [pf[pf["arm"] == a]["name"].iloc[0] for a in order]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    bp = ax.boxplot(data, showfliers=False, widths=0.5, patch_artist=True,
                    medianprops=dict(color=C_TEXT, lw=1.4),
                    boxprops=dict(facecolor="#dce9f8", edgecolor=C_IDEA),
                    whiskerprops=dict(color=C_MUT),
                    capprops=dict(color=C_MUT))
    for i, d in enumerate(data):
        ax.text(i + 1, d.median() + 0.0008, f"{d.median():.4f}",
                ha="center", fontsize=7.5, color=C_TEXT)
    ax.set_xticklabels([f"{a}\n{n}" for a, n in zip(order, names)],
                       fontsize=8)
    ax.set_ylabel("per-factor |mean WF block IC| (book members)",
                  fontsize=10)
    style_ax(ax)
    save(fig, "f11_per_factor_absic.png")


def f12_lasso_selection(ladder):
    d = ladder[ladder["arm"] != "MQ"].dropna(subset=["lasso_book_n_selected"])
    if not len(d):
        return print("skip f12")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(d))
    share = d["lasso_book_n_selected"] / d["lasso_book_n_avail"]
    cols = [family_color(r) for _, r in d.iterrows()]
    ax.bar(x, share, width=0.6, color=cols)
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, r["lasso_book_n_selected"] / r["lasso_book_n_avail"]
                + 0.012,
                f"{r['lasso_book_n_selected']:.0f}/{r['lasso_book_n_avail']:.0f}",
                ha="center", fontsize=8, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(d["name"], fontsize=8.5, rotation=45, ha="right")
    ax.set_ylabel("share of book selected by Lasso\n(mean over blocks)",
                  fontsize=10)
    style_ax(ax)
    save(fig, "f12_lasso_selection.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ladder, preq = load()
    f01_ladder(ladder)
    f02_block_heatmap(ladder, preq)
    f03_preq_vs_lasso(ladder)
    f04_pool_vs_book(ladder)
    f05_grounding_2x2(ladder)
    f06_flip_neff(ladder)
    f07_cost(ladder)
    f08_component_deltas(ladder)
    f09_model_quality(ladder, preq)
    f10_kg_campaign()
    f11_per_factor_dist()
    f12_lasso_selection(ladder)


if __name__ == "__main__":
    main()
