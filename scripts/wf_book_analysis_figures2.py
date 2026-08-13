"""Figures for the RESTRUCTURED WF book report: L4WF baseline deep-dive +
ablation ladder.  Reads wf_book_analysis/derived + raw + wf_arm_analysis_local;
skips panels whose inputs are not there yet (rerun after the local analysis
chain finishes).  Writes figures2/*.{pdf,png}.
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
LOCAL = REPO / "data/comparisons/wf_arm_analysis_local"
FIG = BASE / "figures2"
FIG.mkdir(parents=True, exist_ok=True)

BL = "L4WF_terra_s0"                      # the baseline arm
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY, INK = "#52514e", "#0b0b0b"

LADDER = [
    ("L0WF_gp_s0",            "GP search (no LLM)"),
    ("L1WF_oneshot_terra_s0", "one-shot ideation"),
    ("L1H_terra_s0",          "+ deterministic scoring"),
    ("L1HB_terra_s0",         "+ scoring, 2× seeds"),
    ("L2WF_terra_s0",         "no retrieval, 12 seeds"),
    ("L2WFP_terra_s0",        "no retrieval, seeds matched"),
    ("L4D_terra_s0",          "+ deterministic evolution"),
    ("L4IC_terra_s0",         "IC-only objective, fixed split"),
    ("L4WF_terra_s0",         "+ LLM evolution (baseline)"),
    ("L5WF_terra_s0",         "+ debate"),
    ("L7WF_terra_s0",         "+ memory"),
]

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
traj = pd.read_csv(DER / "pareto_trajectories.csv")
lsel = pd.read_csv(DER / "lasso_selection.csv")
lfreq = pd.read_csv(DER / "lasso_factor_freq.csv")
ladder = pd.read_csv(DER / "ladder_summary.csv").set_index("arm")
preq_bl = pd.read_csv(RAW / BL / "analysis/prequential_record.csv")
preq_bl = preq_bl[(preq_bl.generation >= 11) & (preq_bl.generation <= 20)]

# ------------------------------------------------- B1 baseline factors
d = pf[pf.arm == BL]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
vv = d["ic_wf_blockmean"].dropna()
order = vv.abs().sort_values().reset_index(drop=True)
signs = vv.reindex(vv.abs().sort_values().index).reset_index(drop=True)
ax.bar(range(len(order)), order,
       color=[ORANGE if s > 0 else BLUE for s in signs], width=0.85)
ax.set_xlabel("factor (sorted by |IC|)")
ax.set_ylabel("|walk-forward IC|")
ax.set_title(f"57 factors, median |IC| {order.median():.3f}", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="positive IC"),
                   Patch(color=BLUE, label="negative IC")], fontsize=6.8,
          loc="upper left")
ax = axes[1]
lim = 0.06
ax.plot([-lim, lim], [-lim, lim], color="#c8c7c0", lw=0.9, ls="--")
ax.axhline(0, color=GRAY, lw=0.7)
ax.axvline(0, color=GRAY, lw=0.7)
ax.scatter(d["ic_is_blockmean"], d["ic_wf_blockmean"], s=18, color=ORANGE,
           alpha=0.8, linewidths=0)
r = d["ic_is_blockmean"].corr(d["ic_wf_blockmean"])
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("in-sample block IC")
ax.set_ylabel("walk-forward block IC")
ax.set_title(f"retention, r = {r:.2f}", fontsize=9)
save(fig, "b1_l4_factors")

# ------------------------------------------------- B2 baseline combiners
vals = {
    "Internal LightGBM refit": preq_bl["combined_oos_ic"].mean(),
    "Lasso, PIT refit": lsel[lsel.arm == BL]["ic_oos"].mean(),
    "LightGBM, static": static.query("arm==@BL and model=='lightgbm'")["ic_wf_blockmean"].iloc[0],
    "Lasso, static": static.query("arm==@BL and model=='lasso'")["ic_wf_blockmean"].iloc[0],
    "Ridge, static": static.query("arm==@BL and model=='ridge'")["ic_wf_blockmean"].iloc[0],
}
CB = [ORANGE, BLUE, YELLOW, AQUA, MAGENTA]
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), width_ratios=[1, 1.4])
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
ax.barh(range(len(vals))[::-1], list(vals.values()), color=CB, height=0.62)
ax.set_yticks(range(len(vals))[::-1], list(vals.keys()), fontsize=8)
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_xlabel("mean WF block IC")
ax = axes[1]
blocks = np.arange(11, 21)
pr = preq_bl.set_index("generation")["combined_oos_ic"].reindex(blocks)
la = lsel[lsel.arm == BL].set_index("block")["ic_oos"].reindex(blocks)
gb = static.query("arm==@BL and model=='lightgbm'")
ax.axhline(0, color=GRAY, lw=0.7)
ax.plot(blocks, pr.values, color=ORANGE, lw=1.8, label="internal LightGBM refit")
ax.plot(blocks, la.values, color=BLUE, lw=1.8, label="lasso PIT refit")
ax.plot(blocks, [gb[f"ic_g{b}"].iloc[0] for b in blocks], color=YELLOW,
        lw=1.5, ls="--", label="LightGBM static")
ax.set_xlabel("walk-forward block")
ax.set_ylabel("block IC")
ax.set_xticks([11, 14, 17, 20])
ax.legend(fontsize=6.8, loc="lower left")
save(fig, "b2_l4_combiners")

# ------------------------------------------------- B3 baseline Pareto axes
AXES4 = [("marginal_value", "Marginal value (ΔIC vs. book)"),
         ("independence", "Independence (residual IC)"),
         ("parsimony", "Complexity (AST nodes)"),
         ("structural_novelty", "Structural novelty (AST distance)")]
t = traj[traj.arm == BL].copy()
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True)
for ax, (axis, lab) in zip(axes.ravel(), AXES4):
    tt = t.copy()
    tt[axis] = tt.groupby("genome_id")[axis].ffill()
    v = tt[axis] if axis != "parsimony" else -tt[axis]
    tt = tt.assign(v=v)
    for gid, g in tt.groupby("genome_id"):
        g = g[g.generation >= 1]
        ax.plot(g.generation, g["v"], color=ORANGE, lw=0.5, alpha=0.18)
    m = tt[tt.generation >= 1].groupby("generation")["v"].mean()
    ax.plot(m.index, m.values, color=INK, lw=2.0)
    ax.set_title(lab, fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("generation")
    ax.set_xticks([1, 5, 10, 15, 20])
save(fig, "b3_l4_pareto")

# ------------------------------------------------- B4 baseline lasso
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
ax = axes[0]
dd = lsel[lsel.arm == BL]
ax.bar(dd["block"], dd["n_selected"], color=BLUE, width=0.62, label="selected")
ax.bar(dd["block"], dd["n_avail"] - dd["n_selected"], bottom=dd["n_selected"],
       color="#d9d8d0", width=0.62, label="available, not selected")
ax.set_xlabel("walk-forward block")
ax.set_ylabel("factors")
ax.set_xticks([11, 14, 17, 20])
ax.legend(fontsize=7)
ax = axes[1]
s = lfreq[lfreq.arm == BL]["n_selected"]
counts = [int((s == 0).sum()), int(((s >= 1) & (s <= 3)).sum()),
          int(((s >= 4) & (s <= 6)).sum()), int((s >= 7).sum())]
ax.bar(range(4), counts, color=[GRAY, BLUE, AQUA, ORANGE], width=0.6)
ax.set_xticks(range(4), ["never", "1–3", "4–6", "7–10"])
ax.set_xlabel("blocks in which selected (of 10)")
ax.set_ylabel("factors")
save(fig, "b4_l4_lasso")

# ------------------------------------------------- L1 ablation ladder
fig, ax = plt.subplots(figsize=(6.3, 3.8))
ys = np.arange(len(LADDER))[::-1]
XNOTE = 0.1
for y, (arm, lab) in zip(ys, LADDER):
    if arm not in ladder.index:
        continue
    row = ladder.loc[arm]
    gamed = arm.startswith("L0WF")
    col = "#b3b2a8" if gamed else (ORANGE if arm == BL else BLUE)
    if not pd.isna(row.get("preq_mean_ic")):
        x, se = row["preq_mean_ic"], row["preq_se"]
        if gamed:
            x = 0.088  # clipped: gamed record off-scale
            ax.scatter([x], [y], s=46, color=col, marker=">", zorder=5,
                       clip_on=False)
            ax.annotate("0.144 (metric-gamed)", (x, y), xytext=(-4, -11),
                        textcoords="offset points", fontsize=6.8, color=GRAY,
                        ha="right")
        else:
            ax.errorbar([x], [y], xerr=[se], fmt="o", color=col, ms=6.5,
                        capsize=2.5, lw=1.4, zorder=5)
    if not pd.isna(row.get("pit_best")):
        ax.scatter([row["pit_best"]], [y], s=52, facecolors="none",
                   edgecolors=AQUA, marker="D", lw=1.5, zorder=4)
    note = []
    if not pd.isna(row.get("n_book")):
        note.append(f"{int(row['n_book'])} F.")
    if not pd.isna(row.get("cost_usd")):
        note.append(f"${row['cost_usd']:.0f}")
    ax.text(XNOTE, y, "   ".join(note), fontsize=7, color=GRAY, va="center")
ax.set_yticks(ys, [lab for _, lab in LADDER])
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_xlim(-0.005, 0.118)
ax.set_ylim(-1.6, len(LADDER) - 0.6)
ax.set_xlabel("walk-forward IC (2021–2026)")
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=BLUE, label="honest prequential record ± SE"),
    Line2D([], [], marker="D", ls="", markerfacecolor="none", color=AQUA,
           label="best PIT combiner on the final book"),
], fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
save(fig, "l1_ladder")

# ---------------------------------------------- L3 what improves (baseline)
curve_path = DER / "l4_learning_curve.csv"
gq = [json.loads(l) for l in (RAW / BL / "evolution/gen_quality.jsonl").open()]
preq_all = [json.loads(l) for l in (RAW / BL / "evolution/prequential.jsonl").open()]
fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.9))
fig.subplots_adjust(hspace=0.52, wspace=0.34)
# A: breadth accumulates
ax = axes[0, 0]
gens = [r["generation"] for r in gq]
ax.plot(gens, [r["kept_pool_size"] for r in gq], color=BLUE, lw=1.8,
        label="validated factor pool")
ax.plot(gens, [r["archive_size_total"] for r in gq], color=ORANGE, lw=1.8,
        label="Pareto archive")
ax.set_title("Breadth accumulates", fontsize=9)
ax.set_ylabel("factors")
ax.legend(fontsize=6.5)
# B: fixed-window learning curve
ax = axes[0, 1]
if curve_path.exists() and curve_path.stat().st_size > 10:
    cv = pd.read_csv(curve_path)
    clean = cv[~cv.eval_seen_by_selection]
    seen = cv[cv.generation >= clean.generation.max()]
    ax.plot(clean.generation, clean.ic_fixed_eval, color=ORANGE, lw=1.8,
            marker="o", ms=3.5)
    ax.plot(seen.generation, seen.ic_fixed_eval, color=ORANGE, lw=1.4,
            ls="--", marker="o", ms=3.5)
    ax.axhline(0, color=GRAY, lw=0.7)
    ax.text(0.02, 0.04, "dashed: selection saw part\nof the eval window",
            transform=ax.transAxes, fontsize=6.3, color=GRAY)
ax.set_title("Book@gen g on fixed 2025–26 window", fontsize=9)
ax.set_ylabel("combined IC (ic-weighted)")
# C: honest record stabilises
ax = axes[1, 0]
wf = [(r["generation"], r["combined_oos_ic"]) for r in preq_all
      if r["generation"] >= 11 and r.get("combined_oos_ic") is not None]
gs = [g for g, _ in wf]
vals2 = [v for _, v in wf]
run_mean = np.cumsum(vals2) / np.arange(1, len(vals2) + 1)
ax.bar(gs, vals2, color="#d9d8d0", width=0.6, label="block IC")
ax.plot(gs, run_mean, color=ORANGE, lw=2.0, label="running mean")
ax.axhline(0, color=GRAY, lw=0.7)
ax.set_title("Honest walk-forward record", fontsize=9)
ax.set_xlabel("generation (= traded block)")
ax.set_xticks([11, 14, 17, 20])
ax.legend(fontsize=6.5)
# D: survival is the quality filter
ax = axes[1, 1]
mem = pd.read_csv(DER / "archive_members.csv").query("arm==@BL")
pfb = pf[pf.arm == BL]
jj = mem.merge(pfb, on="factor_id")
jj["cohort"] = pd.cut(jj.admit_generation, [-1, 5, 10, 15, 20],
                      labels=["0–5", "6–10", "11–15", "16–20"])
coh = jj.groupby("cohort", observed=True)["ic_wf_blockmean"].apply(
    lambda s: s.abs().mean())
ax.bar(range(len(coh)), coh.values, color=ORANGE, width=0.6)
ax.set_xticks(range(len(coh)), coh.index)
ax.set_xlabel("admission generation of surviving factors")
ax.set_ylabel("mean |WF IC|")
ax.set_title("Long survival ⇒ quality (re-scoring filter)", fontsize=9)
for ax in (axes[0, 0], axes[0, 1]):
    ax.set_xlabel("generation")
save(fig, "l3_learning")

# ------------------------------------- L2/L3 cross-arm compacts (need phase 1)
have_new = all((LOCAL / a / "per_factor_blocks.csv").exists()
               for a in ["L1H_terra_s0", "L1HB_terra_s0", "L4D_terra_s0"])
if have_new:
    arms2 = ["L1H_terra_s0", "L1HB_terra_s0", "L2WF_terra_s0",
             "L2WFP_terra_s0", "L4D_terra_s0",
             "L4WF_terra_s0", "L5WF_terra_s0", "L7WF_terra_s0"]
    label2 = ["L1H", "L1HB", "L2", "L2P", "L4D", "L4", "L5", "L7"]
    rows = []
    for a in arms2:
        src = LOCAL / a / "per_factor_blocks.csv"
        if not src.exists():
            src = RAW / a / "analysis/per_factor_blocks.csv"
        p = pd.read_csv(src)
        djson = LOCAL / a / "diversity.json"
        if not djson.exists():
            djson = RAW / a / "analysis/diversity.json"
        dv = json.load(djson.open())
        rows.append({"arm": a,
                     "median_ic": p["ic_wf_blockmean"].median(),
                     "median_abs_ic": p["ic_wf_blockmean"].abs().median(),
                     "share_pos": (p["ic_wf_blockmean"] > 0).mean(),
                     "sign_flip": float((np.sign(p["ic_is_blockmean"])
                                         != np.sign(p["ic_wf_blockmean"])).mean()),
                     "n": len(p), "eff_n": dv["effective_n_participation_ratio"],
                     "mean_abs_corr": dv["mean_abs_corr"]})
    cs = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    fig.subplots_adjust(wspace=0.32)
    ax = axes[0]
    cols = [ORANGE if a == BL else BLUE for a in arms2]
    ax.bar(range(len(cs)), cs["median_abs_ic"], color=cols, width=0.6)
    for i, (v, fl) in enumerate(zip(cs["median_abs_ic"], cs["sign_flip"])):
        ax.text(i, v + 0.0004, f"{fl:.0%}", ha="center", fontsize=6.5,
                color=GRAY)
    ax.set_xticks(range(len(cs)), label2, fontsize=7, rotation=20)
    ax.set_ylabel("median per-factor |WF IC|")
    ax.set_title("label: share of factors whose IC sign flips IS→WF",
                 fontsize=7.5, color=GRAY)
    ax = axes[1]
    ax.bar(np.arange(len(cs)) - 0.18, cs["n"], color="#d9d8d0", width=0.32,
           label="book size")
    ax.bar(np.arange(len(cs)) + 0.18, cs["eff_n"], color=AQUA, width=0.32,
           label="effective N")
    ax.set_xticks(range(len(cs)), label2, fontsize=7, rotation=20)
    ax.set_ylabel("factors")
    ax.legend(fontsize=7)
    save(fig, "l2_cross_arm")
    cs.to_csv(DER / "cross_arm_summary.csv", index=False)
else:
    print("phase-1 outputs not complete yet — skipped l2_cross_arm")
