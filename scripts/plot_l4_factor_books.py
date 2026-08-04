"""Figures for the L4 factor-book analysis (reads data/comparisons/l4_factor_analysis/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/comparisons/l4_factor_analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical palette (light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BOOK_COLOR = {"terra": BLUE, "opus": ORANGE, "zoo": AQUA}
BOOK_LABEL = {"terra": "GPT-5.6 Terra — L4 (44 F.)",
              "opus": "Opus 5 — L2-Evolution (18 F.)",
              "zoo": "101 formulaic alphas"}
WIN_COLOR = {"dev": "#a8a7a1", "test": BLUE, "forward": ORANGE}
WIN_LABEL = {"dev": "DEV (IS+VAL, 2010–21)", "test": "TEST (2021–24, nie enthüllt)",
             "forward": "FORWARD (2024–26, außerhalb Panel)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.edgecolor": GRIDC, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "Helvetica Neue",
})

pf = pd.read_csv(OUT / "per_factor_ic.csv")
comb = pd.read_csv(OUT / "combined_book_ic.csv")
corr = pd.read_csv(OUT / "signal_corr_dev.csv", index_col=0)
_nan = corr.isna().mean(axis=1)
_keep = [c for c in corr.columns if _nan[c] <= 0.9]
corr = corr.loc[_keep, _keep]
near = pd.read_csv(OUT / "nearest_zoo_corr.csv")
summary = json.loads((OUT / "summary.json").read_text())
ic_series = pd.read_csv(OUT / "combined_cs_ic_series.csv", index_col=0, parse_dates=True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── 1. dev vs test scatter ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (xw, yw) in zip(axes, [("dev", "test"), ("dev", "forward")]):
    lim = 0.045
    ax.axhline(0, color=GRIDC, lw=0.8)
    ax.axvline(0, color=GRIDC, lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], color=GRIDC, lw=0.8, ls="--", zorder=1)
    for bk in ["zoo", "terra", "opus"]:
        d = pf[pf.book == bk]
        ax.scatter(d[f"ic_pooled_{xw}"], d[f"ic_pooled_{yw}"], s=26 if bk != "zoo" else 16,
                   color=BOOK_COLOR[bk], alpha=0.55 if bk == "zoo" else 0.9,
                   edgecolors="white", linewidths=0.5, label=BOOK_LABEL[bk], zorder=3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"IC {WIN_LABEL[xw].split(' (')[0]} (pooled, h=6)")
    ax.set_ylabel(f"IC {WIN_LABEL[yw].split(' (')[0]}")
    ax.set_title(f"{WIN_LABEL[xw].split(' (')[0]} -> {WIN_LABEL[yw].split(' (')[0]}",
                 fontsize=11, color=INK)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Per-Faktor-IC: In-Search vs. Out-of-Sample", y=1.02, fontsize=13, color=INK)
save(fig, "fig1_ic_scatter.png")

# ── 2. top factors per book ─────────────────────────────────────────────────
for bk in ["terra", "opus"]:
    d = pf[pf.book == bk].dropna(subset=["ic_pooled_dev"]).copy()
    d["abs_dev"] = d.ic_pooled_dev.abs()
    d = d.sort_values("abs_dev", ascending=True).tail(15)
    ylab = [fid[:38] for fid in d.factor_id]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.6))
    h = 0.27
    for k, w in enumerate(["dev", "test", "forward"]):
        ax.barh(y + (1 - k) * h, d[f"ic_pooled_{w}"], height=h * 0.92,
                color=WIN_COLOR[w], label=WIN_LABEL[w])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlabel("pooled per-underlying IC (h=6)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title(f"Top-15 Faktoren nach |DEV-IC| — {BOOK_LABEL[bk]}", fontsize=12, color=INK)
    save(fig, f"fig2_top_factors_{bk}.png")

# ── 3. combined book ICs ────────────────────────────────────────────────────
order = ["zoo_101", "terra_book", "terra_plus_zoo", "opus_book", "opus_plus_zoo"]
labels = {"zoo_101": "101 Alphas\nallein", "terra_book": "Terra-Buch\n(44)",
          "terra_plus_zoo": "Terra + 101\n(145)", "opus_book": "Opus-Buch\n(18)",
          "opus_plus_zoo": "Opus + 101\n(119)"}
comb_i = comb.set_index("combo").loc[[o for o in order if o in set(comb.combo)]]
x = np.arange(len(comb_i))
fig, ax = plt.subplots(figsize=(9.5, 4.6))
w = 0.26
for k, win in enumerate(["dev", "test", "forward"]):
    v = comb_i[f"ic_pooled_{win}"]
    b = ax.bar(x + (k - 1) * w, v, width=w * 0.94, color=WIN_COLOR[win], label=WIN_LABEL[win])
    for xi, vi in zip(x + (k - 1) * w, v):
        if np.isfinite(vi):
            ax.text(xi, vi + (0.001 if vi >= 0 else -0.0035), f"{vi:.3f}",
                    ha="center", fontsize=7.5, color=INK2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([labels[i] for i in comb_i.index], fontsize=9)
ax.set_ylabel("pooled IC des kombinierten Signals (h=6)")
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.set_title("Kombiniertes Buch (LightGBM, Fit auf DEV): IC nach Fenster", fontsize=12, color=INK, pad=34)
save(fig, "fig3_combined_ic.png")

# ── 4. correlation heatmaps per book (clustered) ────────────────────────────
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

fig, axes = plt.subplots(1, 2, figsize=(13, 5.8),
                         gridspec_kw={"width_ratios": [44, 18]})
for ax, bk in zip(axes, ["terra", "opus"]):
    ks = [c for c in corr.columns if c.startswith(bk + ":")]
    sub = corr.loc[ks, ks].fillna(0)
    D = 1 - np.abs(sub.to_numpy()); np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    o = leaves_list(Z)
    S = sub.to_numpy()[np.ix_(o, o)]
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(BOOK_LABEL[bk], fontsize=11, color=INK)
    ax.set_xticks([])
    ax.grid(False)
    if bk == "opus":
        ax.yaxis.tick_right()
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([ks[i].split(":", 1)[1][:34] for i in o], fontsize=6.5)
    else:
        ax.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, label="Signal-Korrelation (DEV, cs-z)")
fig.suptitle("Korrelationsstruktur innerhalb der Bücher (hierarchisch geordnet)",
             fontsize=13, color=INK)
save(fig, "fig4_corr_heatmaps.png")

# ── 5. full corr map with blocks ────────────────────────────────────────────
ks = ([c for c in corr.columns if c.startswith("terra:")]
      + [c for c in corr.columns if c.startswith("opus:")]
      + [c for c in corr.columns if c.startswith("zoo:")])
S = corr.loc[ks, ks].fillna(0).to_numpy()
n_t = sum(c.startswith("terra:") for c in ks)
n_o = sum(c.startswith("opus:") for c in ks)
fig, ax = plt.subplots(figsize=(7.6, 6.8))
im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
for p in (n_t, n_t + n_o):
    ax.axhline(p - 0.5, color=INK, lw=1.0)
    ax.axvline(p - 0.5, color=INK, lw=1.0)
mid = [n_t / 2, n_t + n_o / 2, n_t + n_o + (len(ks) - n_t - n_o) / 2]
ax.set_xticks(mid)
ax.set_xticklabels(["Terra", "Opus", "101 Alphas"], fontsize=10)
ax.set_yticks(mid)
ax.set_yticklabels(["Terra", "Opus", "101 Alphas"], fontsize=10, rotation=90, va="center")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="Signal-Korrelation (DEV)")
ax.set_title("Alle drei Faktormengen: Kreuz-Korrelationsblöcke", fontsize=12, color=INK)
save(fig, "fig5_corr_blocks.png")

# ── 6. novelty vs zoo ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.arange(0, 1.05, 0.05)
for bk in ["terra", "opus"]:
    d = near[near.book == bk]
    axes[0].hist(d.max_abs_corr_zoo, bins=bins, histtype="step", lw=2,
                 color=BOOK_COLOR[bk], label=BOOK_LABEL[bk])
axes[0].set_xlabel("max |Korrelation| zu den 101 formulaic alphas")
axes[0].set_ylabel("# Faktoren")
axes[0].legend(frameon=False, fontsize=8.5)
axes[0].set_title("Neuheit ggü. der formulaic-Bibliothek", fontsize=11, color=INK)
for bk in ["terra", "opus"]:
    d = near[near.book == bk]
    axes[1].scatter(d.max_abs_corr_zoo, d.max_abs_corr_own_book, s=30,
                    color=BOOK_COLOR[bk], alpha=0.85, edgecolors="white", linewidths=0.5,
                    label=BOOK_LABEL[bk])
axes[1].plot([0, 1], [0, 1], color=GRIDC, ls="--", lw=0.8)
axes[1].set_xlabel("max |corr| zum Zoo")
axes[1].set_ylabel("max |corr| im eigenen Buch")
axes[1].set_title("Redundanz: eigenes Buch vs. Zoo", fontsize=11, color=INK)
save(fig, "fig6_zoo_novelty.png")

# ── 7. effective number of factors ──────────────────────────────────────────
sets = [("terra", summary["terra_n"], summary["terra_eff_n"]),
        ("opus", summary["opus_n"], summary["opus_eff_n"]),
        ("zoo", summary["zoo_n"], summary["zoo_eff_n"]),
        ("terra+zoo", None, summary["eff_n_terra+zoo"]),
        ("opus+zoo", None, summary["eff_n_opus+zoo"]),
        ("alle", None, summary["eff_n_terra+opus+zoo"])]
names = [s[0] for s in sets]
tot = [s[1] if s[1] else {"terra+zoo": summary["terra_n"] + summary["zoo_n"],
                          "opus+zoo": summary["opus_n"] + summary["zoo_n"],
                          "alle": summary["terra_n"] + summary["opus_n"] + summary["zoo_n"]}[s[0]]
       for s in sets]
eff = [s[2] for s in sets]
x = np.arange(len(sets))
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.bar(x - 0.2, tot, width=0.38, color="#d9d8d3", label="# Faktoren")
ax.bar(x + 0.2, eff, width=0.38, color=BLUE, label="effektive # (Participation Ratio)")
for xi, (t, e) in zip(x, zip(tot, eff)):
    ax.text(xi - 0.2, t + 1, str(t), ha="center", fontsize=8.5, color=INK2)
    ax.text(xi + 0.2, e + 1, f"{e:.1f}", ha="center", fontsize=8.5, color=BLUE)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Anzahl")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Diversität: nominelle vs. effektive Faktorzahl (DEV-Korrelationen)",
             fontsize=12, color=INK)
save(fig, "fig7_effective_n.png")

# ── 8. rolling CS IC of combined signals ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 4.6))
roll = ic_series[["terra_book", "opus_book", "zoo_101"]].rolling(63, min_periods=30).mean()
wins = summary["windows"]
ax.axvspan(pd.Timestamp(wins["test"][0]), pd.Timestamp(wins["test"][1]),
           color="#f8ac54", alpha=0.05, lw=0)
ax.axvspan(pd.Timestamp(wins["forward"][0]), pd.Timestamp(wins["forward"][1]),
           color="#e34948", alpha=0.05, lw=0)
for c, lb in [("terra_book", "terra"), ("opus_book", "opus"), ("zoo_101", "zoo")]:
    ax.plot(roll.index, roll[c], color=BOOK_COLOR[lb], lw=1.6, label=BOOK_LABEL[lb])
ax.axhline(0, color=INK2, lw=0.8)
for b, lab in [(wins["test"][0], "TEST"), (wins["forward"][0], "FORWARD")]:
    ax.axvline(pd.Timestamp(b), color=INK2, lw=0.9, ls="--")
    ax.text(pd.Timestamp(b), ax.get_ylim()[1], f" {lab}", fontsize=8.5, color=INK2, va="top")
ax.set_ylabel("rolling 63-Tage Ø cross-sectional IC")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Kombiniertes Signal (LightGBM, Fit nur auf DEV): rollierender IC über die Zeit",
             fontsize=12, color=INK)
save(fig, "fig8_rolling_ic.png")

print("all figures ->", FIG)
