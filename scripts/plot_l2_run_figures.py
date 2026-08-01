#!/usr/bin/env python
"""Thesis figure suite for the L2_opus5_s0 evolutionary factor-research run.

Reads the pre-computed ``analysis_data.json`` and renders every figure twice
(PDF for LaTeX inclusion + 300-dpi PNG) into the run's ``figures/`` folder.

Style: no titles/suptitles (LaTeX captions), serif fonts, colorblind-safe
palette (dataviz reference palette, light mode), recessive grid, one rcParams
block shared by every figure.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
# Default is the original L2 run; pass another prerun dir (absolute or
# repo-relative) as argv[1] to render the same suite for any analysed run.
import sys  # noqa: E402

_DEFAULT = "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns/L2_opus5_s0"
RUN_DIR = (Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_absolute()
           else ROOT / (sys.argv[1] if len(sys.argv) > 1 else _DEFAULT))
DATA_PATH = RUN_DIR / "analysis_data.json"
FIG_DIR = RUN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- palette (dataviz)
# Categorical slots, light mode, fixed order (never cycled beyond assignment).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Blue sequential ramp (magnitude / ordered generations).
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_BLUES = LinearSegmentedColormap.from_list("qf_blues", SEQ_BLUES)
# Diverging blue <-> red with a neutral gray midpoint (signed correlations).
CMAP_DIV = LinearSegmentedColormap.from_list(
    "qf_div", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f2321"]
)

CATEGORY_COLORS = {
    "microstructure": BLUE,
    "mean_reversion": ORANGE,
    "statistical_arbitrage": AQUA,
    "sentiment": MAGENTA,
    "volatility": VIOLET,
    "momentum": YELLOW,
    "fundamental": GREEN,
    "carry": "#2a78d6",
    "other": MUTED,
    None: MUTED,
}
CATEGORY_LABELS = {
    "microstructure": "Microstructure",
    "mean_reversion": "Mean reversion",
    "statistical_arbitrage": "Statistical arbitrage",
    "sentiment": "Sentiment / revisions",
    "volatility": "Volatility",
    "momentum": "Momentum",
    "fundamental": "Fundamental",
    "carry": "Carry",
    "other": "Other",
    None: "Uncategorised",
}

# ------------------------------------------------------------------- rcParams (once)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,  # unused (no titles) but kept consistent
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

W = 5.2  # ~13.2 cm thesis column width, inches

saved: list[Path] = []


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)


def trunc(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def zero_line(ax, axis: str = "y", value: float = 0.0, **kw) -> None:
    style = dict(color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    style.update(kw)
    if axis == "y":
        ax.axhline(value, **style)
    else:
        ax.axvline(value, **style)


def fmt_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# --------------------------------------------------------------------------- data
data = json.loads(DATA_PATH.read_text())

archive = data["archive"]
gen_quality = data["gen_quality"]
prequential = data["prequential"]
billed = data["billed"]
evictions_by_gen = {int(k): v for k, v in data["evictions_by_gen"].items()}
rescore_deltas = data["rescore_deltas"]
usage = data["usage"]
factor_meta = data["factor_meta"]
per_factor_fit = data["per_factor_fit"]
combined = data["combined"]
corr_ids = data["corr_matrix"]["ids"]
corr_values = np.asarray(data["corr_matrix"]["values"], dtype=float)
eigenvalues = np.asarray(data["eigenvalues"], dtype=float)

gens = [g["generation"] for g in gen_quality]
n_gens = max(gens) + 1
reveal_gens = sorted({r["generation"] for r in prequential})
preq_ends = [datetime.fromisoformat(r["end"]) for r in prequential]
book_ids = [a["factor_id"] for a in archive]
book_set = set(book_ids)

# ==========================================================================
# 1. prequential_oos_ic
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.0))
zero_line(ax)
ax.plot(preq_ends, [r["combined_oos_ic"] for r in prequential], color=BLUE,
        marker="o", markersize=5.5, markerfacecolor=BLUE,
        markeredgecolor="white", markeredgewidth=0.8)
ax.set_xlabel("Reveal window end")
ax.set_ylabel("Combined prequential OOS IC")
fmt_dates(ax)
save(fig, "prequential_oos_ic")

# ==========================================================================
# 2. prequential_pbo
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
ax.annotate("PBO = 0.5", xy=(preq_ends[0], 0.5), xytext=(0, 4),
            textcoords="offset points", color=MUTED, fontsize=9)
ax.plot(preq_ends, [r["pbo"] for r in prequential], color=ORANGE,
        marker="o", markersize=5.5, markerfacecolor=ORANGE,
        markeredgecolor="white", markeredgewidth=0.8)
ax.set_xlabel("Reveal window end")
ax.set_ylabel("Probability of backtest overfitting")
ax.set_ylim(-0.03, 1.03)
fmt_dates(ax)
save(fig, "prequential_pbo")

# ==========================================================================
# 3. prequential_ic_and_pbo — two stacked panels (single axis each)
# ==========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True,
                               gridspec_kw={"hspace": 0.12})
zero_line(ax1)
ax1.plot(preq_ends, [r["combined_oos_ic"] for r in prequential], color=BLUE,
         marker="o", markersize=5, markerfacecolor=BLUE,
         markeredgecolor="white", markeredgewidth=0.8)
ax1.set_ylabel("Combined OOS IC")
ax2.axhline(0.5, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
ax2.plot(preq_ends, [r["pbo"] for r in prequential], color=ORANGE,
         marker="o", markersize=5, markerfacecolor=ORANGE,
         markeredgecolor="white", markeredgewidth=0.8)
ax2.set_ylabel("PBO")
ax2.set_ylim(-0.03, 1.03)
ax2.set_xlabel("Reveal window end")
fmt_dates(ax2)
save(fig, "prequential_ic_and_pbo")

# ==========================================================================
# 4. archive_kept_pool — two panels sharing the generation axis
# ==========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True,
                               gridspec_kw={"hspace": 0.12, "height_ratios": [3, 2]})
for g in reveal_gens:
    for ax in (ax1, ax2):
        ax.axvline(g, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3)),
                   alpha=0.7, zorder=1)
arch_sizes = [g["archive_size_total"] for g in gen_quality]
ax1.plot(gens, arch_sizes, color=BLUE, marker="o", markersize=4.5,
         markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
ax1.set_ylabel("Archive size")
ax1.set_ylim(0, max(arch_sizes) * 1.28)
# annotate the two big culls (gen 7 and gen 15) in the empty band below the troughs
for g_cull in (7, 15):
    n_ev = evictions_by_gen.get(g_cull, 0)
    y = arch_sizes[g_cull]
    ax1.annotate(f"cull at gen {g_cull}\n({n_ev} evictions)",
                 xy=(g_cull, y), xytext=(g_cull + 0.7, 1.5),
                 fontsize=9, color=INK_2, va="bottom",
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8,
                                 shrinkB=4))
ax1.legend(handles=[Line2D([], [], color=AXIS, linestyle=(0, (4, 3)),
                           linewidth=0.8, label="data-reveal generation")],
           loc="upper left")
ax2.plot(gens, [g["kept_pool_size"] for g in gen_quality], color=ORANGE,
         marker="o", markersize=4.5, markerfacecolor=ORANGE,
         markeredgecolor="white", markeredgewidth=0.7, zorder=3)
ax2.set_ylabel("Kept-pool size")
ax2.set_xlabel("Generation")
ax2.set_xticks(range(0, n_gens, 2))
save(fig, "archive_kept_pool")

# ==========================================================================
# 5. marginal_by_generation
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.1))
zero_line(ax)
ax.plot(gens, [g["mean_marginal_value"] for g in gen_quality], color=BLUE,
        marker="o", markersize=4, label="archive mean")
ax.plot(gens, [g["max_marginal_value"] for g in gen_quality], color=ORANGE,
        marker="s", markersize=4, label="archive max")
ax.set_xlabel("Generation")
ax.set_ylabel("Marginal value (LOCO $\\Delta$IC)")
ax.set_xticks(range(0, n_gens, 2))
ax.legend(loc="upper right", ncols=2)
save(fig, "marginal_by_generation")

# ==========================================================================
# 6. novelty_by_generation
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.plot(gens, [g["mean_structural_novelty"] for g in gen_quality], color=AQUA,
        marker="o", markersize=4.5, markerfacecolor=AQUA,
        markeredgecolor="white", markeredgewidth=0.7)
ax.set_xlabel("Generation")
ax.set_ylabel("Mean structural novelty (archive)")
ax.set_xticks(range(0, n_gens, 2))
ax.set_ylim(0, 1.02)
save(fig, "novelty_by_generation")

# ==========================================================================
# 7. admission_and_evictions
# ==========================================================================
billed_per_gen = Counter(b["generation"] for b in billed)
sel_per_gen = Counter(b["generation"] for b in billed if b["selectable"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True,
                               gridspec_kw={"hspace": 0.12, "height_ratios": [3, 2]})
xs = np.arange(n_gens)
ax1.bar(xs, [billed_per_gen.get(g, 0) for g in xs], width=0.72, color="#9ec5f4",
        edgecolor="white", linewidth=0.5, label="billed candidates")
ax1.plot(xs, [sel_per_gen.get(g, 0) for g in xs], color="#184f95", marker="o",
         markersize=4, linewidth=1.4, label="gate-passing (selectable)")
ax1.set_ylabel("Candidates per generation")
ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2,
           borderaxespad=0.0)
ax2.bar(xs, [evictions_by_gen.get(int(g), 0) for g in xs], width=0.72, color=RED,
        edgecolor="white", linewidth=0.5)
ax2.set_ylabel("Archive evictions")
ax2.set_xlabel("Generation")
ax2.set_xticks(range(0, n_gens, 2))
save(fig, "admission_and_evictions")

# ==========================================================================
# 8. operator_marginal_dist — box plots ordered by median
# ==========================================================================
by_op: dict[str, list[float]] = {}
for b in billed:
    if b["marginal_value"] is not None:
        by_op.setdefault(b["operator"], []).append(b["marginal_value"])
ops_sorted = sorted(by_op, key=lambda o: np.median(by_op[o]))
fig, ax = plt.subplots(figsize=(W, 3.4))
zero_line(ax)
bp = ax.boxplot([by_op[o] for o in ops_sorted], vert=True, widths=0.5,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=2.5, markerfacecolor=MUTED,
                                markeredgecolor="none", alpha=0.55),
                medianprops=dict(color=ORANGE, linewidth=1.6),
                boxprops=dict(facecolor="#cde2fb", edgecolor=INK_2, linewidth=0.9),
                whiskerprops=dict(color=INK_2, linewidth=0.9),
                capprops=dict(color=INK_2, linewidth=0.9))
y0, y1 = ax.get_ylim()
ax.set_ylim(y0, y1 + 0.14 * (y1 - y0))  # headroom for the n= annotations
OP_TICK = {"llm_semantic": "llm\nsemantic", "llm_semantic_creative": "llm semantic\ncreative"}
ax.set_xticks(range(1, len(ops_sorted) + 1))
ax.set_xticklabels([OP_TICK.get(o, o.replace("_", " ")) for o in ops_sorted])
for i, o in enumerate(ops_sorted, start=1):
    ax.annotate(f"n = {len(by_op[o])}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                xytext=(0, -3), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=MUTED)
ax.set_xlabel("Mutation operator")
ax.set_ylabel("Candidate marginal value (LOCO $\\Delta$IC)")
save(fig, "operator_marginal_dist")

# ==========================================================================
# 9. operator_counts
# ==========================================================================
op_billed = Counter(b["operator"] for b in billed)
op_sel = Counter(b["operator"] for b in billed if b["selectable"])
ops_by_n = sorted(op_billed, key=op_billed.get, reverse=True)
fig, ax = plt.subplots(figsize=(W, 3.2))
x = np.arange(len(ops_by_n))
bw = 0.38
b1 = ax.bar(x - bw / 2, [op_billed[o] for o in ops_by_n], width=bw, color=BLUE,
            edgecolor="white", linewidth=0.5, label="billed candidates")
b2 = ax.bar(x + bw / 2, [op_sel[o] for o in ops_by_n], width=bw, color=ORANGE,
            edgecolor="white", linewidth=0.5, label="gate-passing (selectable)")
for bars in (b1, b2):
    ax.bar_label(bars, fontsize=9, color=INK_2, padding=1.5)
ax.set_xticks(x)
ax.set_xticklabels([OP_TICK.get(o, o.replace("_", " ")) for o in ops_by_n])
ax.set_xlabel("Mutation operator")
ax.set_ylabel("Candidates")
ax.set_ylim(0, max(op_billed.values()) * 1.15)
ax.legend(loc="upper right")
save(fig, "operator_counts")

# ==========================================================================
# 10. rescore_drift — histogram + per-generation strip
# ==========================================================================
deltas = np.array([r["delta"] for r in rescore_deltas])
delta_gens = np.array([r["generation"] for r in rescore_deltas])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W, 4.8),
                               gridspec_kw={"hspace": 0.38, "height_ratios": [2, 2]})
ax1.hist(deltas, bins=36, color=BLUE, edgecolor="white", linewidth=0.5)
ax1.axvline(0.0, color=MUTED, linewidth=0.9, linestyle="--")
ax1.axvline(float(np.median(deltas)), color=ORANGE, linewidth=1.4,
            label=f"median = {np.median(deltas):+.4f}")
ax1.set_xlabel("Re-score $\\Delta$ marginal value at data reveal")
ax1.set_ylabel("Archive members")
ax1.legend(loc="upper left")
rng = np.random.default_rng(0)
zero_line(ax2)
ax2.scatter(delta_gens + rng.uniform(-0.22, 0.22, size=len(delta_gens)), deltas,
            s=12, color=BLUE, alpha=0.45, edgecolors="none", zorder=3)
gmeans = [(g, deltas[delta_gens == g].mean()) for g in sorted(set(delta_gens))]
ax2.plot([g for g, _ in gmeans], [m for _, m in gmeans], color=ORANGE, marker="D",
         markersize=4.5, linewidth=1.3, zorder=4, label="per-generation mean")
ax2.set_xlabel("Reveal generation")
ax2.set_ylabel("Re-score $\\Delta$ marginal value")
ax2.set_xticks(sorted(set(delta_gens)))
ax2.legend(loc="lower left")
save(fig, "rescore_drift")

# ------------------------------------------------------------------ book helpers
book_cat = {fid: (factor_meta.get(fid) or {}).get("category") for fid in book_ids}
# None (degenerate/missing) → 0.0 so sorting/limits never see NoneType
loco = {fid: per_factor_fit[fid]["loco_marginal"] or 0.0 for fid in book_ids}
solo = {fid: per_factor_fit[fid]["solo_val_ic"] or 0.0 for fid in book_ids}
_cats_present = [c for c, _ in Counter(book_cat.values()).most_common()]
cat_legend = [Patch(facecolor=CATEGORY_COLORS.get(c, MUTED),
                    label=CATEGORY_LABELS.get(c, str(c)))
              for c in _cats_present]

# ==========================================================================
# 11. book_loco — horizontal bars, sorted, colored by category
# ==========================================================================
order = sorted(book_ids, key=lambda f: loco[f])
fig, ax = plt.subplots(figsize=(W, 4.6))
y = np.arange(len(order))
ax.barh(y, [loco[f] for f in order], height=0.68,
        color=[CATEGORY_COLORS.get(book_cat[f], MUTED) for f in order],
        edgecolor="white", linewidth=0.5)
zero_line(ax, axis="x")
ax.set_yticks(y)
ax.set_yticklabels([trunc(f) for f in order], fontsize=9)
ax.set_xlabel("LOCO marginal contribution ($\\Delta$ combined VAL IC)")
ax.legend(handles=cat_legend, loc="lower right")
ax.grid(axis="y", visible=False)
save(fig, "book_loco")

# ==========================================================================
# 12. book_solo_vs_loco — labeled scatter with quadrant lines
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 4.8))
zero_line(ax, axis="y")
zero_line(ax, axis="x")
for f in book_ids:
    ax.scatter(solo[f], loco[f], s=42, color=CATEGORY_COLORS.get(book_cat[f], MUTED),
               edgecolors="white", linewidths=0.7, zorder=3)
ax.set_xlabel("Solo VAL IC")
ax.set_ylabel("LOCO marginal contribution")
ax.legend(handles=cat_legend, loc="upper left")
xmax = max(abs(v) for v in solo.values()) * 1.5
ax.set_xlim(-xmax, xmax)
# Label placement: sparse points are labeled beside the mark; the dense cluster
# around the origin gets a leader-line stack in the empty left half.
sx = np.array([solo[f] for f in book_ids])
sy = np.array([loco[f] for f in book_ids])
xr, yr = np.ptp(sx), np.ptp(sy)
nx, ny = sx / xr, sy / yr
nn = np.array([min(np.hypot(nx[i] - nx[j], ny[i] - ny[j])
                   for j in range(len(sx)) if j != i) for i in range(len(sx))])
dense = [f for i, f in enumerate(book_ids) if nn[i] < 0.09]
sparse = [f for f in book_ids if f not in dense]
for i, f in enumerate(sorted(sparse, key=lambda k: loco[k])):
    dy = -10 if loco[f] < np.median(sy) else 5
    ax.annotate(trunc(f, 24), xy=(solo[f], loco[f]), xytext=(5, dy),
                textcoords="offset points", fontsize=8, color=INK_2, zorder=4)
ylo, yhi = ax.get_ylim()
stack_y = np.linspace(ylo + 0.12 * (yhi - ylo), yhi - 0.30 * (yhi - ylo), len(dense))
for yt, f in zip(stack_y, sorted(dense, key=lambda k: loco[k])):
    ax.annotate(trunc(f, 24), xy=(solo[f], loco[f]),
                xytext=(-xmax * 0.96, yt), textcoords="data",
                fontsize=8, color=INK_2, va="center", zorder=4,
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.6,
                                shrinkA=2, shrinkB=3, alpha=0.9,
                                relpos=(1.0, 0.5)))
save(fig, "book_solo_vs_loco")

# ==========================================================================
# 13. book_objectives_scatter — marginal vs independence; size=novelty, color=gen
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.8))
mv = np.array([a["objective"]["marginal_value"] for a in archive])
ind = np.array([a["objective"]["independence"] for a in archive])
nov = np.array([a["objective"]["structural_novelty"] for a in archive])
born = np.array([a["generation"] for a in archive])
size = 25 + 220 * (nov - nov.min()) / max(nov.max() - nov.min(), 1e-9)
zero_line(ax, axis="y")
zero_line(ax, axis="x")
sc = ax.scatter(mv, ind, s=size, c=born, cmap=CMAP_BLUES, vmin=0, vmax=n_gens - 1,
                edgecolors=INK_2, linewidths=0.6, alpha=0.9, zorder=3)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Generation admitted")
cb.outline.set_edgecolor(AXIS)
for q, lab in ((nov.min(), f"novelty {nov.min():.2f}"), (nov.max(), f"novelty {nov.max():.2f}")):
    ax.scatter([], [], s=25 + 220 * (q - nov.min()) / max(nov.max() - nov.min(), 1e-9),
               facecolors="none", edgecolors=INK_2, linewidths=0.7, label=lab)
ax.legend(loc="upper right", labelspacing=1.0, borderpad=0.7)
ax.set_xlabel("Marginal value (LOCO $\\Delta$IC)")
ax.set_ylabel("Independence (residual IC)")
save(fig, "book_objectives_scatter")

# ==========================================================================
# 14. corr_heatmap — 18x18 diverging, centered at 0
# ==========================================================================
vmax = np.ceil(np.abs(corr_values[~np.eye(len(corr_ids), dtype=bool)]).max() * 10) / 10
fig, ax = plt.subplots(figsize=(W + 0.6, W + 0.2))
im = ax.imshow(corr_values, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax)
n = len(corr_ids)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
labels = [trunc(f, 26) for f in corr_ids]
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.grid(visible=False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
# 18x18 cells are too small for legible >=8pt in-cell values -> no annotation
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("Signal correlation")
cb.outline.set_edgecolor(AXIS)
save(fig, "corr_heatmap")

# ==========================================================================
# 15. eigen_spectrum — variance shares + cumulative, PR annotated
# ==========================================================================
lam = eigenvalues
share = lam / lam.sum()
cum = np.cumsum(share)
pr = (lam.sum() ** 2) / (lam ** 2).sum()
fig, ax = plt.subplots(figsize=(W, 3.4))
comp = np.arange(1, len(lam) + 1)
ax.bar(comp, share, width=0.7, color=BLUE, edgecolor="white", linewidth=0.5,
       label="variance share $\\lambda_i/\\sum\\lambda$")
ax.plot(comp, cum, color=ORANGE, marker="o", markersize=3.5, linewidth=1.5,
        label="cumulative share")
ax.axhline(1.0 / len(lam), color=MUTED, linewidth=0.9, linestyle="--",
           label=f"equal-weight (1/{len(lam)})")
ax.axvline(pr, color=AQUA, linewidth=1.3, linestyle=(0, (4, 3)))
ax.annotate(f"participation ratio\n$(\\sum\\lambda)^2/\\sum\\lambda^2 = {pr:.1f}$",
            xy=(pr, 0.30), xytext=(pr - 0.5, 0.30), fontsize=9, color=INK_2,
            ha="right", va="center")
ax.set_xlabel("Eigenvalue rank")
ax.set_ylabel("Share of total variance")
ax.set_xticks(range(2, len(lam) + 1, 2))
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left")
save(fig, "eigen_spectrum")

# ==========================================================================
# 16. book_age_hist — generation born of final-book members
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 2.9))
counts = np.bincount([a["generation"] for a in archive], minlength=n_gens)
ax.bar(range(n_gens), counts, width=0.72, color=BLUE, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Generation admitted")
ax.set_ylabel("Final-book members")
ax.set_xticks(range(0, n_gens, 2))
ax.set_yticks(range(0, counts.max() + 1, 2))
save(fig, "book_age_hist")

# ==========================================================================
# 17. book_category_mix
# ==========================================================================
cat_counts = Counter(book_cat.values())
cats = sorted(cat_counts, key=cat_counts.get, reverse=True)
fig, ax = plt.subplots(figsize=(W, 2.6))
bars = ax.barh(np.arange(len(cats)), [cat_counts[c] for c in cats], height=0.6,
               color=[CATEGORY_COLORS.get(c, MUTED) for c in cats], edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fontsize=9.5, color=INK_2, padding=3)
ax.set_yticks(np.arange(len(cats)))
ax.set_yticklabels([CATEGORY_LABELS.get(c, str(c).title()) for c in cats])
ax.invert_yaxis()
ax.set_xlabel("Final-book members")
ax.set_xlim(0, max(cat_counts.values()) * 1.15)
ax.grid(axis="y", visible=False)
save(fig, "book_category_mix")

# ==========================================================================
# 18. cost_by_role — cost panel + token panel
# ==========================================================================
roles = sorted(usage["by_role"], key=lambda r: usage["by_role"][r]["cost_usd"], reverse=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True,
                               gridspec_kw={"wspace": 0.12})
y = np.arange(len(roles))
costs = [usage["by_role"][r]["cost_usd"] for r in roles]
bars = ax1.barh(y, costs, height=0.62, color=BLUE, edgecolor="white", linewidth=0.5)
ax1.bar_label(bars, fmt="$%.0f", fontsize=9, color=INK_2, padding=3)
ax1.set_yticks(y)
ax1.set_yticklabels([
    f"{r} ({usage['by_role'][r]['calls']} call{'s' if usage['by_role'][r]['calls'] != 1 else ''})"
    for r in roles])
ax1.invert_yaxis()
ax1.set_xlabel("LLM cost (USD)")
ax1.set_xlim(0, max(costs) * 1.22)
ax1.grid(axis="y", visible=False)
bh = 0.32
ax2.barh(y - bh / 2, [usage["by_role"][r]["input_tokens"] / 1e6 for r in roles],
         height=bh, color=BLUE, edgecolor="white", linewidth=0.5, label="input")
ax2.barh(y + bh / 2, [usage["by_role"][r]["output_tokens"] / 1e6 for r in roles],
         height=bh, color=ORANGE, edgecolor="white", linewidth=0.5, label="output")
ax2.set_xlabel("Tokens (millions)")
ax2.legend(loc="lower right")
ax2.grid(axis="y", visible=False)
save(fig, "cost_by_role")

# ==========================================================================
# 19. candidate_cloud — all billed candidates, book highlighted
# ==========================================================================
cand = [b for b in billed if b["structural_novelty"] is not None
        and b["marginal_value"] is not None]
cn = np.array([b["structural_novelty"] for b in cand])
cm = np.array([b["marginal_value"] for b in cand])
cg = np.array([b["generation"] for b in cand])
fig, ax = plt.subplots(figsize=(W, 3.8))
zero_line(ax)
sc = ax.scatter(cn, cm, s=11, c=cg, cmap=CMAP_BLUES, vmin=0, vmax=n_gens - 1,
                alpha=0.45, edgecolors="none", zorder=2)
bn = [a["objective"]["structural_novelty"] for a in archive]
bm = [a["objective"]["marginal_value"] for a in archive]
ax.scatter(bn, bm, s=55, marker="D", facecolor=ORANGE, edgecolors=INK,
           linewidths=0.8, zorder=4, label=f"final book (n = {len(archive)})")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Generation")
cb.outline.set_edgecolor(AXIS)
ax.set_xlabel("Structural novelty")
ax.set_ylabel("Marginal value (LOCO $\\Delta$IC)")
ax.legend(loc="lower left")
save(fig, "candidate_cloud")

# ==========================================================================
# 20. per_factor_prequential — per-reveal IC of final-book factors
# ==========================================================================
present = {f for r in prequential for f in r["per_factor_ic"] if f in book_set}
top3 = sorted(present, key=lambda f: loco[f], reverse=True)[:3]
top_colors = {top3[0]: BLUE, top3[1]: ORANGE, top3[2]: AQUA}
fig, ax = plt.subplots(figsize=(W, 3.6))
zero_line(ax)
for fid in book_ids:
    pts = [(preq_ends[i], r["per_factor_ic"][fid])
           for i, r in enumerate(prequential) if fid in r["per_factor_ic"]]
    if not pts:
        continue
    xs_, ys_ = zip(*pts)
    if fid in top_colors:
        ax.plot(xs_, ys_, color=top_colors[fid], linewidth=1.9, marker="o",
                markersize=4.5, markerfacecolor=top_colors[fid],
                markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label=trunc(fid, 26))
    else:
        ax.plot(xs_, ys_, color=MUTED, linewidth=0.9, alpha=0.5, marker="o",
                markersize=2.5, zorder=2)
handles, labels_ = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color=MUTED, linewidth=0.9, alpha=0.6,
                      label="other book members"))
ax.legend(handles=handles + [], loc="upper left", fontsize=8.5)
ax.set_xlabel("Reveal window end")
ax.set_ylabel("Per-factor prequential OOS IC")
fmt_dates(ax)
save(fig, "per_factor_prequential")

# ==========================================================================
# 21. (extra) book_solo_ic_combined — solo ICs vs combined-model VAL IC
# ==========================================================================
order = sorted(book_ids, key=lambda f: solo[f])
fig, ax = plt.subplots(figsize=(W, 4.6))
y = np.arange(len(order))
ax.barh(y, [solo[f] for f in order], height=0.68,
        color=[CATEGORY_COLORS.get(book_cat[f], MUTED) for f in order],
        edgecolor="white", linewidth=0.5)
zero_line(ax, axis="x")
ax.axvline(combined["ridge_val_ic"], color=VIOLET, linewidth=1.5, linestyle=(0, (4, 2)))
ax.axvline(combined["lightgbm_val_ic"], color=INK, linewidth=1.5, linestyle=(0, (1, 2)))
ax.set_yticks(y)
ax.set_yticklabels([trunc(f) for f in order], fontsize=9)
ax.set_xlabel("VAL IC")
ax.legend(handles=cat_legend +
          [Line2D([], [], color=VIOLET, linewidth=1.5, linestyle=(0, (4, 2)),
                  label=f"combined ridge ({combined['ridge_val_ic']:.3f})"),
           Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (1, 2)),
                  label=f"combined LightGBM ({combined['lightgbm_val_ic']:.3f})")],
          loc="lower right", bbox_to_anchor=(0.76, 0.02), fontsize=8.5)
ax.grid(axis="y", visible=False)
save(fig, "book_solo_ic_combined")

# ==========================================================================
# 22. (extra) book_complexity_vs_novelty — parsimony/novelty trade-off
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.4))
for f, a in zip(book_ids, archive):
    ax.scatter(a["diagnostics"]["complexity"], a["objective"]["structural_novelty"],
               s=46, color=CATEGORY_COLORS.get(book_cat[f], MUTED), edgecolors="white",
               linewidths=0.7, zorder=3)
ax.set_xlabel("AST complexity (nodes)")
ax.set_ylabel("Structural novelty")
ax.legend(handles=cat_legend, loc="lower right")
save(fig, "book_complexity_vs_novelty")

# --------------------------------------------------------------------------- report
print(f"Wrote {len(saved)} files to {FIG_DIR}")
for p in sorted(FIG_DIR.iterdir()):
    print(f"  {p.name:45s} {p.stat().st_size / 1024:8.1f} KB")
