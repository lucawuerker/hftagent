"""Figures for the combiner-model study (reads data/comparisons/combiner_models/)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/comparisons/combiner_models"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# fixed categorical order (validated palette, light mode)
MODEL_COLOR = {"ridge": "#2a78d6", "lasso": "#eb6834", "lightgbm": "#1baf7a",
               "lgbm_l2n": "#eda100", "lgbm_l2": "#e87ba4"}
MODEL_LABEL = {"ridge": "Ridge", "lasso": "Lasso", "lightgbm": "LightGBM",
               "lgbm_l2n": "LightGBM L2-lambda=N", "lgbm_l2": "LightGBM L2-lambda=10N"}
MODELS = ["ridge", "lasso", "lightgbm", "lgbm_l2n", "lgbm_l2"]
SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.edgecolor": GRIDC, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Helvetica Neue",
})

fw = pd.json_normalize([json.loads(l) for l in open(OUT / "combiner_forward.jsonl")])
wf = pd.json_normalize([json.loads(l) for l in open(OUT / "combiner_wf.jsonl")])
# WF panel on the per-block metric (mean IC per 126-bar block, IS and OOS alike)
wfb = pd.read_csv(OUT / "combined_wf_blocks.csv")
wfb = wfb.rename(columns={"ic_fit_blockmean": "ic_fit", "ic_wf_blockmean": "ic_wf"})

BOOK_FW = {"terra": "Terra L4 (44)", "opus": "Opus L2 (18)", "zoo": "101 alphas",
           "terra_plus_opus": "Terra+Opus (62)", "terra_plus_opus_zoo": "alle (163)"}
BOOK_WF = {"l2wf": "L2WF (19)", "l4wf": "L4WF (57)", "zoo": "101 alphas",
           "l2wf_plus_l4wf": "L2WF+L4WF (76)"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── fig 1: dumbbell fit -> OOS ──────────────────────────────────────────────
def dumbbell(ax, df, books, oos_col, title):
    rows = []
    for bk in books:
        for m in MODELS:
            r = df[(df.combo == bk) & (df.model == m)]
            if len(r):
                rows.append((bk, m, float(r.ic_fit.iloc[0]), float(r[oos_col].iloc[0])))
    y = 0
    yticks, ylabels = [], []
    for i, (bk, m, fit, oos) in enumerate(rows):
        if i > 0 and rows[i - 1][0] != bk:
            y -= 1  # gap between books
        c = MODEL_COLOR[m]
        ax.plot([oos, fit], [y, y], color=c, lw=1.6, alpha=0.55, zorder=2)
        ax.scatter([fit], [y], facecolors="white", edgecolors=c, s=34, zorder=3, lw=1.6)
        ax.scatter([oos], [y], color=c, s=38, zorder=4, edgecolors="white", lw=0.6)
        yticks.append(y)
        ylabels.append(f"{books[bk]}  ·  {MODEL_LABEL[m]}")
        y -= 1
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("pooled IC (h=6)")
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlim(-0.005, 0.26)


fig, axes = plt.subplots(1, 2, figsize=(13, 7.6))
dumbbell(axes[0], fw, BOOK_FW, "ic_test",
         "Forward-Panel: Fit (DEV, offener Kreis) -> TEST (gefüllt)")
dumbbell(axes[1], wfb, BOOK_WF, "ic_wf",
         "WF-Panel (Server): IS-Blockmittel (offen) -> WF-Blockmittel 2021–26 (gefüllt)")
handles = [plt.Line2D([], [], color=MODEL_COLOR[m], marker="o", ls="-",
                      label=MODEL_LABEL[m]) for m in MODELS]
axes[0].legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("Der In-Sample->OOS-Kollaps nach Kombinationsmodell", fontsize=13, color=INK)
fig.tight_layout()
save(fig, "fig1_dumbbell.png")

# ── fig 2: OOS bars ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
for ax, (df, books, col, title) in zip(axes, [
        (fw, BOOK_FW, "ic_test", "Forward-Panel: TEST 2021–24"),
        (fw, BOOK_FW, "ic_forward", "Forward-Panel: FORWARD 2024–26"),
        (wfb, BOOK_WF, "ic_wf", "WF-Panel: ø IC je WF-Block 2021–26")]):
    combos = list(books)
    x = np.arange(len(combos))
    w = 0.16
    for k, m in enumerate(MODELS):
        v = [float(df[(df.combo == c) & (df.model == m)][col].iloc[0]) for c in combos]
        ax.bar(x + (k - 2) * w, v, width=w * 0.92, color=MODEL_COLOR[m],
               label=MODEL_LABEL[m])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([books[c] for c in combos], fontsize=8, rotation=12)
    ax.set_title(title, fontsize=10.5, color=INK)
axes[0].set_ylabel("OOS pooled IC (h=6)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:5], labels[:5], frameon=False, fontsize=8.5, ncol=5,
           loc="upper center", bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Out-of-sample-IC nach Buch und Modell", fontsize=13, color=INK, y=1.02)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig2_oos_bars.png")

# ── fig 3: WF per-block vs prequential ──────────────────────────────────────
WSP = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
for ax, (arm, combo) in zip(axes, [("L4WF_terra_s0", "l4wf"), ("L2WF_terra_s0", "l2wf")]):
    preq = [json.loads(l) for l in open(WSP / arm / "evolution/prequential.jsonl")]
    preq = [r for r in preq if r["generation"] >= 11]
    ends = [pd.Timestamp(r["end"]) for r in preq]
    ax.plot(ends, [r["combined_oos_ic"] for r in preq], color=INK, lw=1.8, marker="o",
            ms=5, label="Prequential-Record des Laufs (Refit je Generation)")
    for m in ["ridge", "lasso", "lightgbm"]:
        row = wf[(wf.combo == combo) & (wf.model == m)].iloc[0]
        bl = {k.split(".")[1]: row[k] for k in wf.columns
              if k.startswith("blocks.") and k.endswith(".ic")}
        # columns are blocks.block_gN.ic
        xs, ys = [], []
        for g in range(11, 21):
            key = f"blocks.block_g{g}.ic"
            if key in wf.columns:
                xs.append(pd.Timestamp(row[f"blocks.block_g{g}.end"]))
                ys.append(row[key])
        ax.plot(xs, ys, color=MODEL_COLOR[m], lw=1.5, marker="o", ms=4,
                label=f"statischer Fit (->2021-07): {MODEL_LABEL[m]}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(f"{arm} — kombinierter IC je Walk-forward-Block", fontsize=11, color=INK)
    ax.set_ylabel("pooled IC im Block")
axes[0].legend(frameon=False, fontsize=8.5, ncol=2)
fig.tight_layout()
save(fig, "fig3_blocks.png")

# ── fig 4: retention anatomy ────────────────────────────────────────────────
pf_fw = pd.read_csv(OUT / "per_factor_forward.csv")
pf_wf = pd.read_csv(OUT / "per_factor_blockmetric_wf.csv")
entries = []
for bk, label in [("terra", "Terra L4"), ("opus", "Opus L2")]:
    d = pf_fw[pf_fw.book == bk].dropna(subset=["ic_fit"])
    perf = d.ic_test.abs().mean() / d.ic_fit.abs().mean()
    row = {"book": label, "perfactor": perf}
    for m in ["ridge", "lasso", "lightgbm"]:
        r = fw[(fw.combo == bk) & (fw.model == m)].iloc[0]
        row[m] = r.ic_test / r.ic_fit
    entries.append(row)
for bk, label in [("l2wf", "L2WF"), ("l4wf", "L4WF")]:
    d = pf_wf[pf_wf.book == bk].dropna(subset=["ic_fit_blockmean"])
    perf = d.ic_wf_blockmean.abs().mean() / d.ic_fit_blockmean.abs().mean()
    row = {"book": label, "perfactor": perf}
    for m in ["ridge", "lasso", "lightgbm"]:
        r = wfb[(wfb.combo == bk) & (wfb.model == m)].iloc[0]
        row[m] = r.ic_wf / r.ic_fit
    entries.append(row)
E = pd.DataFrame(entries)
x = np.arange(len(E))
w = 0.19
fig, ax = plt.subplots(figsize=(9.6, 4.4))
series = [("perfactor", "ø Einzelfaktor (mean |IC|)", "#a8a7a1"),
          ("ridge", "Ridge kombiniert", MODEL_COLOR["ridge"]),
          ("lasso", "Lasso kombiniert", MODEL_COLOR["lasso"]),
          ("lightgbm", "LightGBM kombiniert", MODEL_COLOR["lightgbm"])]
for k, (colname, lab, c) in enumerate(series):
    v = E[colname] * 100
    ax.bar(x + (k - 1.5) * w, v, width=w * 0.92, color=c, label=lab)
    for xi, vi in zip(x + (k - 1.5) * w, v):
        ax.text(xi, vi + 2, f"{vi:.0f}%", ha="center", fontsize=7.5, color=INK2)
ax.axhline(100, color=INK2, lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(E.book)
ax.set_ylabel("OOS-Retention (OOS-IC / Fit-IC, %)")
ax.legend(frameon=False, fontsize=8.5)
ax.set_title("Anatomie des Kollapses: Einzelfaktoren halten, der GBM-Fit kollabiert",
             fontsize=12, color=INK)
fig.tight_layout()
save(fig, "fig4_retention.png")

# ── fig 5: lasso sparsity ───────────────────────────────────────────────────
rows = []
for df, books in [(fw, BOOK_FW), (wf, BOOK_WF)]:
    for c, lab in books.items():
        r = df[(df.combo == c) & (df.model == "lasso")].iloc[0]
        if (c, lab) not in [x[:2] for x in rows]:
            rows.append((c, lab, int(r.n_factors), int(r["diag.n_nonzero_coef"])))
seen = set()
rows = [r for r in rows if not (r[1] in seen or seen.add(r[1]))]
labs = [r[1] for r in rows]
y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.barh(y + 0.18, [r[2] for r in rows], height=0.34, color="#d9d8d3",
        label="Faktoren im Buch")
ax.barh(y - 0.18, [r[3] for r in rows], height=0.34, color=MODEL_COLOR["lasso"],
        label="von Lasso genutzt (Koeff. ≠ 0)")
for yi, r in zip(y, rows):
    ax.text(r[2] + 1.5, yi + 0.18, str(r[2]), va="center", fontsize=8, color=INK2)
    ax.text(r[3] + 1.5, yi - 0.18, str(r[3]), va="center", fontsize=8,
            color=MODEL_COLOR["lasso"])
ax.set_yticks(y)
ax.set_yticklabels(labs, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("# Faktoren")
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title("Lasso wählt radikal aus: genutzte vs. vorhandene Faktoren",
             fontsize=12, color=INK)
fig.tight_layout()
save(fig, "fig5_lasso_sparsity.png")

print("figures ->", FIG)
