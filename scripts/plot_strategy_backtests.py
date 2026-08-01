#!/usr/bin/env python
"""Thesis figure suite for the six catalog_nasdaq100_v1 strategy backtests.

Reads ``backtest_viz_data.json`` and renders every figure twice (PDF for LaTeX
inclusion + 300-dpi PNG) into ``figures_backtests/``.

Style matches ``scripts/plot_l2_run_figures.py``: no titles/suptitles (LaTeX
captions carry those), serif fonts, colorblind-safe categorical palette
(dataviz reference palette, light mode, fixed slot order), recessive grid.

CRITICAL SEMANTICS encoded on every time-series figure: the plotted equity is
produced by the FINAL model refit on all data before ``oos_start`` — so both
the IS and VAL calendar segments are in-sample for this artifact (VAL was the
model-selection window); only the segment from ``oos_start`` onward is
genuinely out-of-sample.  Background shading + dashed boundaries + in-band
tags make that unmistakable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "data/books/catalog_nasdaq100_v1"

_parser = argparse.ArgumentParser(description="Render the strategy-backtest figure suite.")
_parser.add_argument("--input", type=Path, default=BOOK_DIR / "backtest_viz_data.json",
                     help="viz-bundle JSON (default: the cross-sectional bundle)")
_parser.add_argument("--outdir", type=Path, default=BOOK_DIR / "figures_backtests",
                     help="output figure directory (default: figures_backtests/)")
_args = _parser.parse_args()
DATA_PATH = _args.input
FIG_DIR = _args.outdir
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- palette (dataviz)
# Categorical slots, light mode, fixed order (validated: all six checks pass;
# the sub-3:1 contrast WARN is relieved by legends/direct labels everywhere).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
RED = "#e34948"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Segment shading: IS very light neutral, VAL slightly darker neutral, OOS white.
IS_SHADE = "#f4f3ee"
VAL_SHADE = "#e8e6dd"

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


def fmt_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


# --------------------------------------------------------------------------- data
data = json.loads(DATA_PATH.read_text())

VAL_START = pd.Timestamp(data["val_start"])
OOS_START = pd.Timestamp(data["oos_start"])
COST_RATE_BP = data["cost_rate_bp"]

# Fixed strategy order + fixed palette-slot assignment (never cycled).
STRATS = [
    "combined_book",
    "aggressive_short_term",
    "defensive_low_turnover",
    "earnings_events",
    "diversified_all_weather",
    "contrarian_dip_buyer",
]
COLORS = {
    "combined_book": BLUE,
    "aggressive_short_term": ORANGE,
    "defensive_low_turnover": AQUA,
    "earnings_events": YELLOW,
    "diversified_all_weather": MAGENTA,
    "contrarian_dip_buyer": GREEN,
}
LABELS = {
    "combined_book": "Combined book",
    "aggressive_short_term": "Aggressive short-term",
    "defensive_low_turnover": "Defensive low-turnover",
    "earnings_events": "Earnings events",
    "diversified_all_weather": "All-weather",
    "contrarian_dip_buyer": "Contrarian dip-buyer",
}

strategies = data["strategies"]
missing = [s for s in STRATS if s not in strategies]
assert not missing, f"strategies missing from JSON: {missing}"

# Net daily return series (nulls dropped) and derived equity / drawdown.
returns: dict[str, pd.Series] = {}
equity: dict[str, pd.Series] = {}
drawdown: dict[str, pd.Series] = {}
for s in STRATS:
    r = pd.Series(strategies[s]["returns"], dtype=float)
    r.index = pd.to_datetime(r.index)
    r = r.sort_index().dropna()
    eq = (1.0 + r).cumprod()
    returns[s] = r
    equity[s] = eq
    drawdown[s] = eq / eq.cummax() - 1.0

X0 = min(r.index[0] for r in returns.values())
X1 = max(r.index[-1] for r in returns.values())

ANN = 252  # daily bars


def shade_segments(ax, tags: bool = True, tag_fs: float = 7.5) -> None:
    """IS / VAL / OOS background convention (see module docstring)."""
    ax.axvspan(X0, VAL_START, color=IS_SHADE, zorder=0)
    ax.axvspan(VAL_START, OOS_START, color=VAL_SHADE, zorder=0)
    for x in (VAL_START, OOS_START):
        ax.axvline(x, color=MUTED, linewidth=0.9, linestyle="--", zorder=2)
    if tags:
        trans = ax.get_xaxis_transform()
        for x0, x1, label in (
            (X0, VAL_START, "fit (in-sample)"),
            (VAL_START, OOS_START, "selection\n(in-sample)"),
            (OOS_START, X1, "held-out"),
        ):
            xm = x0 + (x1 - x0) / 2
            ax.text(xm, 0.975, label, transform=trans, ha="center", va="top",
                    fontsize=tag_fs, color=INK_2, zorder=5, linespacing=1.15)


def zero_line(ax, axis: str = "y", value: float = 0.0, **kw) -> None:
    style = dict(color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    style.update(kw)
    if axis == "y":
        ax.axhline(value, **style)
    else:
        ax.axvline(value, **style)


# ==========================================================================
# 1. bt_<name> — per-strategy equity + drawdown (6 figures)
# ==========================================================================
for s in STRATS:
    c = COLORS[s]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(W, 4.4), sharex=True,
        gridspec_kw={"hspace": 0.10, "height_ratios": [2.4, 1.0]})
    shade_segments(ax1, tags=True)
    shade_segments(ax2, tags=False)
    eq = equity[s]
    ax1.plot(eq.index, eq.values, color=c, linewidth=1.4, zorder=3)
    ax1.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
    ax1.set_ylabel("Cumulative net equity")
    # headroom so the band tags never collide with the curve
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 + 0.10 * (y1 - y0))
    dd = drawdown[s] * 100.0
    ax2.fill_between(dd.index, dd.values, 0.0, color=c, alpha=0.35,
                     linewidth=0.0, zorder=3)
    ax2.plot(dd.index, dd.values, color=c, linewidth=0.8, zorder=4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.set_xlim(X0, X1)
    fmt_dates(ax2)
    save(fig, f"bt_{s}")

# ==========================================================================
# 2. bt_all_oos_rebased — all six, OOS only, rebased to 1.0 at oos_start
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.4))
ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
for s in STRATS:
    eq = equity[s].loc[equity[s].index >= OOS_START]
    eq = eq / eq.iloc[0]
    ax.plot(eq.index, eq.values, color=COLORS[s], linewidth=1.4,
            label=LABELS[s], zorder=3)
ax.set_xlabel("Date (held-out period)")
ax.set_ylabel("Net equity (rebased at OOS start)")
ax.legend(loc="lower left", ncols=2, fontsize=8)
fmt_dates(ax)
save(fig, "bt_all_oos_rebased")

# ==========================================================================
# 3. bt_all_full_log — all six full-period curves, log scale, shared shading
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.6))
shade_segments(ax, tags=True)
for s in STRATS:
    eq = equity[s]
    ax.plot(eq.index, eq.values, color=COLORS[s], linewidth=1.2,
            label=LABELS[s], zorder=3)
ax.set_yscale("log")
ax.axhline(1.0, color=AXIS, linewidth=0.8, zorder=1)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative net equity (log scale)")
ax.set_xlim(X0, X1)
# headroom for the band tags
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo, yhi * (yhi / ylo) ** 0.10)
# legend outside above the axes so it never collides with the band tags
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=3, fontsize=8,
          columnspacing=1.1, handlelength=1.6, borderaxespad=0.0)
fmt_dates(ax)
save(fig, "bt_all_full_log")

# ==========================================================================
# 4. bt_segment_sharpe — grouped bars per strategy; OOS saturated, IS/VAL muted
# ==========================================================================
SEG_COLORS = {"IS": "#d8d6cc", "VAL": "#b3b0a3", "OOS": BLUE}
SEG_LABELS = {"IS": "IS (in-sample)", "VAL": "VAL (selection, in-sample)",
              "OOS": "OOS (held-out)"}
fig, ax = plt.subplots(figsize=(W + 0.8, 3.2))
x = np.arange(len(STRATS))
bw = 0.26
zero_line(ax)
for i, seg in enumerate(("IS", "VAL", "OOS")):
    vals = [strategies[s]["metrics"][seg]["sharpe"] for s in STRATS]
    ax.bar(x + (i - 1) * bw, vals, width=bw - 0.02, color=SEG_COLORS[seg],
           edgecolor="white", linewidth=0.5, label=SEG_LABELS[seg],
           zorder=3 if seg == "OOS" else 2)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[s].replace(" ", "\n", 1) for s in STRATS], fontsize=7.5)
ax.set_ylabel("Annualised Sharpe ratio (net)")
y0, y1 = ax.get_ylim()
ax.set_ylim(y0, y1 + 0.16 * (y1 - y0))  # headroom for the legend
ax.legend(loc="upper right", fontsize=8)
save(fig, "bt_segment_sharpe")

# ==========================================================================
# 5. bt_oos_metrics — OOS Sharpe | forward IC (+ gross-Sharpe dot panel)
# ==========================================================================
order = sorted(STRATS, key=lambda s: strategies[s]["metrics"]["OOS"]["sharpe"])
y = np.arange(len(order))
have_gross = all(strategies[s].get("forward_gross_sharpe") is not None for s in order)
ncols = 3 if have_gross else 2
widths = [3, 2, 2][:ncols]
fig, axes = plt.subplots(1, ncols, figsize=(W + 1.2, 2.9), sharey=True,
                         gridspec_kw={"wspace": 0.12, "width_ratios": widths})
ax1, ax2 = axes[0], axes[1]
ax1.barh(y, [strategies[s]["metrics"]["OOS"]["sharpe"] for s in order],
         height=0.62, color=[COLORS[s] for s in order],
         edgecolor="white", linewidth=0.5, zorder=3)
zero_line(ax1, axis="x")
ax1.set_yticks(y)
ax1.set_yticklabels([LABELS[s] for s in order], fontsize=9)
ax1.set_xlabel("OOS Sharpe (net)")
ax1.grid(axis="y", visible=False)
ax2.barh(y, [strategies[s]["forward_ic"] for s in order], height=0.62,
         color=[COLORS[s] for s in order], edgecolor="white",
         linewidth=0.5, zorder=3)
zero_line(ax2, axis="x")
ax2.set_xlabel("Forward IC")
ax2.grid(axis="y", visible=False)
if have_gross:
    # Gross Sharpe lives on a Sharpe scale, not an IC scale, so it gets its
    # own panel (single-axis rule) instead of markers inside the IC panel.
    ax3 = axes[2]
    zero_line(ax3, axis="x")
    ax3.scatter([strategies[s]["forward_gross_sharpe"] for s in order], y,
                s=34, color=[COLORS[s] for s in order], edgecolors="white",
                linewidths=0.7, zorder=3)
    ax3.set_xlabel("Forward gross Sharpe")
    ax3.grid(axis="y", visible=False)
    ax3.xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
for a in axes:
    a.tick_params(axis="x", labelsize=8)
save(fig, "bt_oos_metrics")

# ==========================================================================
# 6. bt_cost_drag — cost drag bars + OOS net / approx-gross return dumbbells
# ==========================================================================
order = sorted(STRATS, key=lambda s: strategies[s]["metrics"]["OOS"]["ann_ret"])
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(W, 3.2))
zero_line(ax, axis="x")
drag = [strategies[s]["ann_cost_drag"] for s in order]
net = [strategies[s]["metrics"]["OOS"]["ann_ret"] for s in order]
gross = [n + d for n, d in zip(net, drag)]
ax.barh(y, drag, height=0.55, color="#f2b8b7", edgecolor="white",
        linewidth=0.5, zorder=2)
for yi, n, g in zip(y, net, gross):
    ax.plot([n, g], [yi, yi], color=INK_2, linewidth=1.0, zorder=3)
ax.scatter(net, y, s=38, color=BLUE, edgecolors="white", linewidths=0.8, zorder=4)
ax.scatter(gross, y, s=38, facecolors="white", edgecolors=INK_2,
           linewidths=1.1, zorder=4)
ax.set_yticks(y)
ax.set_yticklabels([LABELS[s] for s in order], fontsize=9)
ax.set_xlabel("Annualised return (fraction)")
ax.grid(axis="y", visible=False)
ax.legend(handles=[
    Patch(facecolor="#f2b8b7", edgecolor="white",
          label=f"cost drag ({COST_RATE_BP} bp)"),
    Line2D([], [], marker="o", color="none", markerfacecolor=BLUE,
           markeredgecolor="white", markersize=6.5, label="OOS net ann. return"),
    Line2D([], [], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=INK_2, markersize=6.5,
           label="net + drag ($\\approx$ gross)"),
], loc="upper left", fontsize=8)
save(fig, "bt_cost_drag")

# ==========================================================================
# 7. bt_rolling_sharpe — 126-day rolling annualised Sharpe, all six
# ==========================================================================
fig, ax = plt.subplots(figsize=(W, 3.6))
shade_segments(ax, tags=True)
zero_line(ax)
for s in STRATS:
    r = returns[s]
    rs = r.rolling(126).mean() / r.rolling(126).std() * np.sqrt(ANN)
    ax.plot(rs.index, rs.values, color=COLORS[s], linewidth=1.0,
            label=LABELS[s], zorder=3)
ax.set_xlabel("Date")
ax.set_ylabel("Rolling 126-day Sharpe (annualised, net)")
ax.set_xlim(X0, X1)
ylo, yhi = ax.get_ylim()
ax.set_ylim(ylo, yhi + 0.12 * (yhi - ylo))  # headroom for the band tags
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=3, fontsize=8,
          columnspacing=1.1, handlelength=1.6, borderaxespad=0.0)
fmt_dates(ax)
save(fig, "bt_rolling_sharpe")

# --------------------------------------------------------------------------- report
print(f"Wrote {len(saved)} files to {FIG_DIR}")
for p in sorted(FIG_DIR.iterdir()):
    print(f"  {p.name:38s} {p.stat().st_size / 1024:8.1f} KB")
