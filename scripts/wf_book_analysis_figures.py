"""Figures for the WF-ladder factor-book analysis (L2/L4/L5/L7).

Reads data/comparisons/wf_book_analysis/derived/ + raw/, writes
data/comparisons/wf_book_analysis/figures/*.{pdf,png}.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/comparisons/wf_book_analysis"
DER = BASE / "derived"
RAW = BASE / "raw"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ARMS = ["L2WF_terra_s0", "L4WF_terra_s0", "L5WF_terra_s0", "L7WF_terra_s0"]
LABEL = {"L2WF_terra_s0": "L2 (no retrieval)", "L4WF_terra_s0": "L4 (GraphRAG)",
         "L5WF_terra_s0": "L5 (debate)", "L7WF_terra_s0": "L7 (memory)"}
# validated categorical palette (light mode), fixed slot order
C = {"L2WF_terra_s0": "#2a78d6", "L4WF_terra_s0": "#eb6834",
     "L5WF_terra_s0": "#1baf7a", "L7WF_terra_s0": "#eda100"}
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


pf = pd.read_csv(DER / "per_factor_summary.csv")
static = pd.read_csv(DER / "combined_static_all.csv")
div = pd.read_csv(DER / "diversity_all.csv")
traj = pd.read_csv(DER / "pareto_trajectories.csv")
members = pd.read_csv(DER / "archive_members.csv")
lsel = pd.read_csv(DER / "lasso_selection.csv")
lfreq = pd.read_csv(DER / "lasso_factor_freq.csv")

preq = {}
for a in ARMS:
    r = pd.read_csv(RAW / a / "analysis/prequential_record.csv")
    preq[a] = r[(r.generation >= 11) & (r.generation <= 20)]

# ---------------------------------------------------------------- F1 strip
fig, ax = plt.subplots(figsize=(6.3, 2.6))
rng = np.random.default_rng(0)
for i, a in enumerate(ARMS):
    v = pf[pf.arm == a]["ic_wf_blockmean"].dropna()
    y = i + rng.uniform(-0.16, 0.16, len(v))
    ax.scatter(v, y, s=16, color=C[a], alpha=0.75, linewidths=0)
    ax.plot([v.median()] * 2, [i - 0.28, i + 0.28], color="#0b0b0b", lw=1.6, zorder=5)
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_yticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
ax.set_xlabel("Per-factor walk-forward IC (mean over the 10 held-out blocks)")
ax.invert_yaxis()
save(fig, "f1_factor_ic_strip")

# ---------------------------------------------------------- F2 retention
fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.4), sharex=True, sharey=True)
for ax, a in zip(axes.ravel(), ARMS):
    d = pf[pf.arm == a]
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.axvline(0, color=GRAY, lw=0.7)
    lim = 0.06
    ax.plot([-lim, lim], [-lim, lim], color="#c8c7c0", lw=0.9, ls="--")
    ax.scatter(d["ic_is_blockmean"], d["ic_wf_blockmean"], s=18, color=C[a],
               alpha=0.8, linewidths=0)
    ax.set_title(LABEL[a], fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
for ax in axes[1]:
    ax.set_xlabel("in-sample block IC (fit window)")
for ax in axes[:, 0]:
    ax.set_ylabel("walk-forward block IC")
save(fig, "f2_ic_retention")

# ------------------------------------------------- F3 combined-book bars
methods = [("prequential", "Internal LightGBM refit (prequential)"),
           ("pit_lasso", "Lasso, PIT refit per block"),
           ("lightgbm", "LightGBM, static fit"),
           ("lasso", "Lasso, static fit"),
           ("ridge", "Ridge, static fit")]
COLM = {"prequential": ORANGE, "pit_lasso": BLUE, "lightgbm": YELLOW,
        "lasso": AQUA, "ridge": MAGENTA}
vals = {}
for a in ARMS:
    vals[a] = {
        "prequential": preq[a]["combined_oos_ic"].mean(),
        "pit_lasso": lsel[lsel.arm == a]["ic_oos"].mean(),
        **{m: static[(static.arm == a) & (static.model == m)]["ic_wf_blockmean"].iloc[0]
           for m in ("lightgbm", "lasso", "ridge")},
    }
fig, ax = plt.subplots(figsize=(6.3, 3.0))
x = np.arange(len(ARMS))
w = 0.15
for j, (m, lab) in enumerate(methods):
    ax.bar(x + (j - 2) * w, [vals[a][m] for a in ARMS], width=w * 0.88,
           color=COLM[m], label=lab)
ax.set_xticks(x, [LABEL[a] for a in ARMS])
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_ylabel("Combined-book WF IC\n(mean over blocks)")
ax.set_ylim(0, 0.082)
ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "f3_combined_book")

# ------------------------------------------- F4 per-block combined lines
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True, sharey=True)
for ax, a in zip(axes.ravel(), ARMS):
    blocks = np.arange(11, 21)
    pr = preq[a].set_index("generation")["combined_oos_ic"].reindex(blocks)
    la = lsel[lsel.arm == a].set_index("block")["ic_oos"].reindex(blocks)
    gb = static[(static.arm == a) & (static.model == "lightgbm")]
    gbv = [gb[f"ic_g{b}"].iloc[0] for b in blocks]
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.plot(blocks, pr.values, color=ORANGE, lw=1.8, label="internal LightGBM refit")
    ax.plot(blocks, la.values, color=BLUE, lw=1.8, label="lasso PIT refit")
    ax.plot(blocks, gbv, color=YELLOW, lw=1.5, ls="--", label="LightGBM static")
    ax.set_title(LABEL[a], fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7.5, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.985))
for ax in axes[1]:
    ax.set_xlabel("walk-forward block (≈6 months each)")
    ax.set_xticks([11, 14, 17, 20])
for ax in axes[:, 0]:
    ax.set_ylabel("block IC")
save(fig, "f4_blocks")

# --------------------------------------------- F5 Pareto axis evolution
AXES4 = [("marginal_value", "Marginal value (ΔIC vs. book)"),
         ("independence", "Independence (residual IC)"),
         ("parsimony", "Complexity (AST nodes)"),
         ("structural_novelty", "Structural novelty (AST distance)")]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8), sharex=True)
for ax, (axis, lab) in zip(axes.ravel(), AXES4):
    for a in ARMS:
        if axis == "independence" and a in ("L5WF_terra_s0", "L7WF_terra_s0"):
            continue  # axis inactive in the debate arms (low-coverage book member)
        t = traj[traj.arm == a].copy()
        # rescores don't always recompute every axis: carry the last known
        # value forward per genome before averaging across the archive
        t[axis] = t.groupby("genome_id")[axis].ffill()
        v = t[axis] if axis != "parsimony" else -t[axis]
        g = t.assign(v=v).groupby("generation")["v"].mean()
        g = g[g.index >= 1]
        ax.plot(g.index, g.values, color=C[a], lw=1.8, label=LABEL[a])
    if axis == "independence":
        ax.text(0.97, 0.95, "L5/L7: axis inactive\n(low-coverage book member)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=GRAY)
    ax.set_title(lab, fontsize=9)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7.5, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, 0.985))
for ax in axes[1]:
    ax.set_xlabel("generation")
    ax.set_xticks([1, 5, 10, 15, 20])
save(fig, "f5_pareto_axes")

# ------------------------------------------------ F6 diversity bars
fig, ax = plt.subplots(figsize=(6.3, 2.6))
x = np.arange(len(ARMS))
ax.bar(x - 0.18, div.set_index("arm").loc[ARMS, "n_factors"], width=0.32,
       color=BLUE, label="book size")
ax.bar(x + 0.18, div.set_index("arm").loc[ARMS, "effective_n_participation_ratio"],
       width=0.32, color=AQUA, label="effective N (participation ratio)")
for i, a in enumerate(ARMS):
    r = div.set_index("arm").loc[a]
    ax.text(i, max(r["n_factors"], r["effective_n_participation_ratio"]) + 1.2,
            f"mean |ρ| = {r['mean_abs_corr']:.3f}", ha="center", fontsize=7.5,
            color=GRAY)
ax.set_xticks(x, [LABEL[a] for a in ARMS])
ax.set_ylabel("factors")
ax.set_ylim(0, 68)
ax.legend(fontsize=8)
save(fig, "f6_diversity")

# ------------------------------------------------ F7 lasso sparsity
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
ax = axes[0]
for a in ARMS:
    d = lsel[lsel.arm == a]
    ax.plot(d["block"], d["n_selected"] / d["n_avail"], color=C[a], lw=1.8,
            label=LABEL[a])
ax.set_xlabel("walk-forward block")
ax.set_ylabel("share of factors\nselected by lasso")
ax.set_ylim(0, 1)
ax.set_xticks([11, 14, 17, 20])
ax.legend(fontsize=6.5, loc="lower right")
ax = axes[1]
buckets = [("never", lambda s: s == 0), ("1–3", lambda s: (s >= 1) & (s <= 3)),
           ("4–6", lambda s: (s >= 4) & (s <= 6)), ("7–10", lambda s: s >= 7)]
BC = [GRAY, BLUE, AQUA, ORANGE]
bottom = np.zeros(len(ARMS))
for (blab, cond), col in zip(buckets, BC):
    share = []
    for a in ARMS:
        s = lfreq[lfreq.arm == a]["n_selected"]
        share.append(cond(s).mean())
    ax.bar(range(len(ARMS)), share, bottom=bottom, color=col, width=0.6,
           label=f"{blab}")
    bottom += np.array(share)
ax.set_xticks(range(len(ARMS)), ["L2", "L4", "L5", "L7"])
ax.set_ylabel("share of book")
ax.legend(fontsize=6.5, title="blocks selected", title_fontsize=6.5,
          loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "f7_lasso_sparsity")

# ------------------------------------------- F8 winner's curse shrinkage
adm = traj[traj.event == "admit"].set_index(["arm", "genome_id"])["marginal_value"]
fin = traj.sort_values("generation").groupby(["arm", "genome_id"]).last()["marginal_value"]
fig, ax = plt.subplots(figsize=(4.6, 4.2))
lim = 0.045
ax.plot([-0.01, lim], [-0.01, lim], color="#c8c7c0", lw=0.9, ls="--")
ax.axhline(0, color=GRAY, lw=0.7)
ax.axvline(0, color=GRAY, lw=0.7)
for a in ARMS:
    x = adm.loc[a]
    y = fin.loc[a].reindex(x.index)
    ax.scatter(x, y, s=16, color=C[a], alpha=0.75, linewidths=0, label=LABEL[a])
means = {a: (adm.loc[a].mean(), fin.loc[a].mean()) for a in ARMS}
for a in ARMS:
    ax.scatter(*means[a], s=90, color=C[a], edgecolors="#0b0b0b", zorder=5)
ax.set_xlabel("marginal value at admission")
ax.set_ylabel("marginal value at final rescore")
ax.legend(fontsize=7)
ax.set_xlim(-0.012, lim)
ax.set_ylim(-0.02, lim)
save(fig, "f8_winners_curse")

# --------------------------------------------- F9 survivor operator mix
OPS = ["seed", "llm_semantic", "llm_semantic_creative", "crossover",
       "cross_group", "jitter"]
OPLAB = {"seed": "seed idea", "llm_semantic": "LLM mutation",
         "llm_semantic_creative": "creative mutation", "crossover": "crossover",
         "cross_group": "cross-group synthesis", "jitter": "window jitter"}
OPC = [GRAY, BLUE, AQUA, ORANGE, MAGENTA, YELLOW]
fig, ax = plt.subplots(figsize=(6.3, 2.6))
bottom = np.zeros(len(ARMS))
for op, col in zip(OPS, OPC):
    share = [float((members[members.arm == a]["operator"] == op).mean())
             for a in ARMS]
    ax.bar(range(len(ARMS)), share, bottom=bottom, color=col, width=0.6,
           label=OPLAB[op])
    bottom += np.array(share)
ax.set_xticks(range(len(ARMS)), [LABEL[a] for a in ARMS])
ax.set_ylabel("share of final archive")
ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "f9_operator_mix")

# summary numbers for the report
summary = {
    "per_factor": {a: {"n": int((pf.arm == a).sum()),
                       "median_wf_ic": float(pf[pf.arm == a]["ic_wf_blockmean"].median()),
                       "share_positive": float((pf[pf.arm == a]["ic_wf_blockmean"] > 0).mean())}
                   for a in ARMS},
    "combined": vals,
    "shrinkage": {a: {"admit_mean": float(means[a][0]), "final_mean": float(means[a][1])}
                  for a in ARMS},
    "lasso": {a: {"mean_selected": float(lsel[lsel.arm == a]["n_selected"].mean()),
                  "mean_avail": float(lsel[lsel.arm == a]["n_avail"].mean()),
                  "never_share": float((lfreq[lfreq.arm == a]["n_selected"] == 0).mean())}
              for a in ARMS},
}
(DER / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
