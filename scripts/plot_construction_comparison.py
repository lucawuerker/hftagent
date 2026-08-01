#!/usr/bin/env python
"""Construction-comparison figures: cross-sectional vs per-underlying.

Compares the six catalog_nasdaq100_v1 strategies under the two
position-construction regimes, reading BOTH viz bundles
(``backtest_viz_data.json`` = cross-sectional, ``backtest_viz_data_pu.json``
= per-underlying) and writing into ``figures_backtests_pu/``:

* ``cmp_oos_equity_<name>``  — OOS-window net equity rebased to 1.0 at the
  OOS start, one line per construction (6 figures).
* ``cmp_oos_sharpe``         — paired horizontal bars, OOS net Sharpe per
  strategy under each construction.
* ``cmp_cost_vs_gross``      — per-strategy dumbbells (gross -> net OOS
  Sharpe) for both constructions: per-underlying reaches a higher gross
  Sharpe but also pays a larger cost drag.

Style matches ``scripts/plot_strategy_backtests.py`` exactly (serif, dataviz
reference palette, recessive grid, no titles; PDF + 300-dpi PNG).  All three
figure families live entirely in the held-out window, so the IS/VAL band
shading is not applicable here.
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

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
BOOK_DIR = ROOT / "data/books/catalog_nasdaq100_v1"

_parser = argparse.ArgumentParser(description="Construction-comparison figures.")
_parser.add_argument("--input-cs", type=Path, default=BOOK_DIR / "backtest_viz_data.json",
                     help="cross-sectional viz bundle")
_parser.add_argument("--input-pu", type=Path, default=BOOK_DIR / "backtest_viz_data_pu.json",
                     help="per-underlying viz bundle")
_parser.add_argument("--outdir", type=Path, default=BOOK_DIR / "figures_backtests_pu",
                     help="output figure directory")
_args = _parser.parse_args()
FIG_DIR = _args.outdir
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- palette (dataviz)
# Same categorical slots as plot_strategy_backtests.py.  The two constructions
# take the first two slots in fixed order: cross-sectional = blue,
# per-underlying = orange (identity, never cycled).
BLUE = "#2a78d6"
ORANGE = "#eb6834"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CONSTRUCTIONS = ("cs", "pu")
CON_COLORS = {"cs": BLUE, "pu": ORANGE}
CON_LABELS = {"cs": "Cross-sectional", "pu": "Per-underlying"}

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


def zero_line(ax, axis: str = "y", value: float = 0.0, **kw) -> None:
    style = dict(color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
    style.update(kw)
    if axis == "y":
        ax.axhline(value, **style)
    else:
        ax.axvline(value, **style)


# --------------------------------------------------------------------------- data
bundles = {
    "cs": json.loads(_args.input_cs.read_text()),
    "pu": json.loads(_args.input_pu.read_text()),
}
assert bundles["cs"]["oos_start"] == bundles["pu"]["oos_start"], "OOS starts differ"
OOS_START = pd.Timestamp(bundles["cs"]["oos_start"])

STRATS = [
    "combined_book",
    "aggressive_short_term",
    "defensive_low_turnover",
    "earnings_events",
    "diversified_all_weather",
    "contrarian_dip_buyer",
]
LABELS = {
    "combined_book": "Combined book",
    "aggressive_short_term": "Aggressive short-term",
    "defensive_low_turnover": "Defensive low-turnover",
    "earnings_events": "Earnings events",
    "diversified_all_weather": "All-weather",
    "contrarian_dip_buyer": "Contrarian dip-buyer",
}
for con in CONSTRUCTIONS:
    missing = [s for s in STRATS if s not in bundles[con]["strategies"]]
    assert not missing, f"strategies missing from {con} bundle: {missing}"

# OOS-window net equity, rebased to 1.0 at OOS start.
oos_equity: dict[str, dict[str, pd.Series]] = {c: {} for c in CONSTRUCTIONS}
for con in CONSTRUCTIONS:
    for s in STRATS:
        r = pd.Series(bundles[con]["strategies"][s]["returns"], dtype=float)
        r.index = pd.to_datetime(r.index)
        r = r.sort_index().dropna()
        r = r.loc[r.index >= OOS_START]
        oos_equity[con][s] = (1.0 + r).cumprod()


def metric(con: str, s: str, *keys):
    node = bundles[con]["strategies"][s]
    for k in keys:
        node = node[k]
    return node


# ==========================================================================
# 1. cmp_oos_equity_<name> — OOS equity, both constructions (6 figures)
# ==========================================================================
for s in STRATS:
    fig, ax = plt.subplots(figsize=(W, 2.9))
    zero_line(ax, value=1.0)  # zero-growth reference
    for con in CONSTRUCTIONS:
        eq = oos_equity[con][s]
        ax.plot(eq.index, eq.values, color=CON_COLORS[con], linewidth=1.4,
                label=CON_LABELS[con], zorder=3)
    ax.set_xlabel("Date (held-out period)")
    ax.set_ylabel("Net equity (rebased at OOS start)")
    ax.legend(loc="best", fontsize=8)
    fmt_dates(ax)
    save(fig, f"cmp_oos_equity_{s}")

# ==========================================================================
# 2. cmp_oos_sharpe — paired horizontal bars, OOS net Sharpe per strategy
# ==========================================================================
order = sorted(STRATS, key=lambda s: np.mean([metric(c, s, "metrics", "OOS", "sharpe")
                                              for c in CONSTRUCTIONS]))
y = np.arange(len(order))
bh = 0.34
fig, ax = plt.subplots(figsize=(W, 3.4))
zero_line(ax, axis="x")
for i, con in enumerate(CONSTRUCTIONS):
    vals = [metric(con, s, "metrics", "OOS", "sharpe") for s in order]
    ax.barh(y + (bh / 2 if con == "cs" else -bh / 2),  # cs above, pu below
            vals, height=bh - 0.04, color=CON_COLORS[con],
            edgecolor="white", linewidth=0.5, label=CON_LABELS[con], zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([LABELS[s] for s in order], fontsize=9)
ax.set_xlabel("OOS Sharpe (net)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=8)
save(fig, "cmp_oos_sharpe")

# ==========================================================================
# 3. cmp_cost_vs_gross — gross -> net OOS Sharpe dumbbells, both constructions
# ==========================================================================
# One row per strategy, one dumbbell per construction (offset within the row).
# Open marker = gross Sharpe, filled marker = net Sharpe; the connector length
# is the cost drag on the Sharpe scale.  Per-underlying sits higher on gross
# but its longer connectors show the larger cost drag it pays.
order = sorted(STRATS, key=lambda s: metric("pu", s, "forward_gross_sharpe"))
y = np.arange(len(order))
off = {"cs": +0.18, "pu": -0.18}
fig, ax = plt.subplots(figsize=(W, 3.6))
zero_line(ax, axis="x")
for con in CONSTRUCTIONS:
    c = CON_COLORS[con]
    for yi, s in zip(y, order):
        g = metric(con, s, "forward_gross_sharpe")
        n = metric(con, s, "metrics", "OOS", "sharpe")
        yy = yi + off[con]
        ax.plot([n, g], [yy, yy], color=c, linewidth=1.3, zorder=3, alpha=0.85)
        ax.scatter([g], [yy], s=34, facecolors="white", edgecolors=c,
                   linewidths=1.2, zorder=4)
        ax.scatter([n], [yy], s=38, color=c, edgecolors="white",
                   linewidths=0.8, zorder=5)
ax.set_yticks(y)
ax.set_yticklabels([LABELS[s] for s in order], fontsize=9)
ax.set_xlabel("OOS Sharpe (annualised): gross $\\rightarrow$ net")
ax.grid(axis="y", visible=False)
handles = [
    Line2D([], [], color=CON_COLORS["cs"], linewidth=1.6, label=CON_LABELS["cs"]),
    Line2D([], [], color=CON_COLORS["pu"], linewidth=1.6, label=CON_LABELS["pu"]),
    Line2D([], [], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=INK_2, markersize=6.5, label="gross"),
    Line2D([], [], marker="o", color="none", markerfacecolor=INK_2,
           markeredgecolor="white", markersize=6.5, label="net"),
]
ax.legend(handles=handles, loc="lower right", fontsize=8, ncols=2)
save(fig, "cmp_cost_vs_gross")

# --------------------------------------------------------------------------- report
print(f"Wrote {len(saved)} files to {FIG_DIR}")
for p in sorted(saved):
    print(f"  {p.name:38s} {p.stat().st_size / 1024:8.1f} KB")
