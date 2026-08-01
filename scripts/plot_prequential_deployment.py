#!/usr/bin/env python
"""Figures for a prequential-deployment walk-forward track (any prerun).

Reads <run_dir>/prequential_deployment/{report.json, stitched_net_*.csv} and
renders (PDF + 300-dpi PNG, same style as plot_l2_run_figures.py):

  prequential_ic_track   — pooled IC per traded segment (incl. TEST + forward)
  prequential_equity     — stitched cumulative net return, both constructions

Usage: ./venv/bin/python scripts/plot_prequential_deployment.py \
           data/workspaces/<ws>/preruns/<run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute()
           else ROOT / sys.argv[1])
DEP = RUN_DIR / "prequential_deployment"

BLUE, ORANGE, MUTED, GRID, AXIS = "#2a78d6", "#eb6834", "#898781", "#e1e0d9", "#c3c2b7"
INK, INK_2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})
W = 5.2


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(DEP / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


report = json.loads((DEP / "report.json").read_text())
segments = report["segments"]

# ── 1. IC track ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.axhline(0, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
for s in segments:
    x0, x1 = pd.Timestamp(s["eval_start"]), pd.Timestamp(s["eval_end"])
    color = {"test_tail": ORANGE, "forward_reserve": "#e34948"}.get(
        s["segment"], BLUE)
    ax.plot([x0, x1], [s["pooled_ic"]] * 2, color=color, linewidth=2.6,
            solid_capstyle="butt")
ax.set_xlabel("Traded window")
ax.set_ylabel("Pooled OOS IC (per segment)")
from matplotlib.lines import Line2D

ax.legend(handles=[
    Line2D([], [], color=BLUE, lw=2.6, label="Reveal blocks (walk-forward)"),
    Line2D([], [], color=ORANGE, lw=2.6, label="TEST tail"),
    Line2D([], [], color="#e34948", lw=2.6, label="Forward reserve"),
], loc="upper right")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
save(fig, "prequential_ic_track")

# ── 2. stitched equity ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(W, 3.2))
for csv, color, label in (
        ("stitched_net_cross_sectional.csv", BLUE, "Cross-sectional"),
        ("stitched_net_per_underlying.csv", ORANGE, "Per-underlying")):
    ser = pd.read_csv(DEP / csv, index_col=0, parse_dates=True).iloc[:, 0]
    ax.plot(ser.index, (1 + ser).cumprod(), color=color, label=label)
ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="--", zorder=1)
# segment boundaries: TEST + forward starts
for s in segments:
    if s["segment"] in ("test_tail", "forward_reserve"):
        ax.axvline(pd.Timestamp(s["eval_start"]), color=MUTED,
                   linewidth=0.9, linestyle=":", zorder=1)
ax.set_xlabel("Date")
ax.set_ylabel("Stitched net equity (×)")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
save(fig, "prequential_equity")

print(f"wrote figures into {DEP}")
