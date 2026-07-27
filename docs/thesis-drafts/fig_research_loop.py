#!/usr/bin/env python
"""Render the "One generation of the framework" research-loop figure.

This is a matplotlib fallback / preview for ``fig_research_loop.tex``. The TikZ
version is the primary thesis artifact (vector, matches document fonts); this
script reproduces the same diagram and writes both a PDF (vector, for the
thesis) and a PNG (quick preview).

    ./venv/bin/python docs/thesis-drafts/fig_research_loop.py

Outputs next to this file: fig_research_loop.pdf and fig_research_loop.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- palette (mirrors the TikZ \definecolor lines) ---------------------------
INK = "#212530"
AGENT = "#2D5FAF"
GOLD = "#B8861C"
GREEN = "#1C805C"
GREY = "#EAECF0"

AGENT_FILL = "#E6ECF7"
GOLD_FILL = "#F6ECD3"
GREEN_FILL = "#DCEEE7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "svg.fonttype": "none",
    }
)

# Role -> (facecolor, edgecolor, linewidth)
ROLE = {
    "agent": (AGENT_FILL, AGENT, 1.1),
    "det": (GREY, "#6B7280", 1.0),
    "state": (GOLD_FILL, GOLD, 1.1),
    "eval": (GREEN_FILL, GREEN, 1.6),
}


def box(ax, cx, cy, w, h, text, role, fontsize=9, weight="normal"):
    fc, ec, lw = ROLE[role]
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        weight=weight,
        zorder=4,
        linespacing=1.25,
    )
    return {"cx": cx, "cy": cy, "w": w, "h": h}


def anchor(node, side):
    cx, cy, w, h = node["cx"], node["cy"], node["w"], node["h"]
    return {
        "N": (cx, cy + h / 2),
        "S": (cx, cy - h / 2),
        "E": (cx + w / 2, cy),
        "W": (cx - w / 2, cy),
        "C": (cx, cy),
    }[side]


def arrow(ax, p0, p1, style="flow", rad=0.0, connector="arc3"):
    styles = {
        "flow": dict(color=INK, lw=1.3, ls="-"),
        "data": dict(color=GOLD, lw=1.4, ls=(0, (3.2, 1.8))),
        "teacher": dict(color=AGENT, lw=1.5, ls=(0, (5, 2.4))),
    }
    s = styles[style]
    a = FancyArrowPatch(
        p0,
        p1,
        connectionstyle=f"{connector},rad={rad}",
        arrowstyle="-|>",
        mutation_scale=13,
        color=s["color"],
        lw=s["lw"],
        linestyle=s["ls"],
        shrinkA=1.5,
        shrinkB=3.0,
        zorder=2,
        capstyle="round",
    )
    ax.add_patch(a)


def elbow(ax, pts, style="flow"):
    """Orthogonal multi-segment arrow through waypoints; arrowhead on last leg."""
    styles = {
        "flow": dict(color=INK, lw=1.3, ls="-"),
        "data": dict(color=GOLD, lw=1.4, ls=(0, (3.2, 1.8))),
        "teacher": dict(color=AGENT, lw=1.5, ls=(0, (5, 2.4))),
    }
    s = styles[style]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=s["color"], lw=s["lw"], ls=s["ls"],
            solid_capstyle="round", zorder=2)
    # last leg carries the arrowhead
    a = FancyArrowPatch(
        pts[-2],
        pts[-1],
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>",
        mutation_scale=13,
        color=s["color"],
        lw=s["lw"],
        linestyle=s["ls"],
        shrinkA=0,
        shrinkB=3.0,
        zorder=2,
    )
    ax.add_patch(a)


def group_frame(ax, x0, y0, x1, y1, edge, fill=None, pad=0.18, ls="-", lw=1.0):
    patch = FancyBboxPatch(
        (x0 - pad, y0 - pad),
        (x1 - x0) + 2 * pad,
        (y1 - y0) + 2 * pad,
        boxstyle="round,pad=0.01,rounding_size=0.12",
        facecolor=fill if fill else "none",
        edgecolor=edge,
        linewidth=lw,
        linestyle=ls,
        zorder=1,
    )
    ax.add_patch(patch)


def elabel(ax, x, y, text, color=INK, bg="white"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.0,
        color=color,
        zorder=5,
        linespacing=1.15,
        bbox=dict(boxstyle="round,pad=0.18", fc=bg, ec="none"),
    )


def build():
    fig, ax = plt.subplots(figsize=(12.4, 9.4))
    ax.set_xlim(-0.9, 15.0)
    ax.set_ylim(-2.55, 10.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Entry path I: ideation ------------------------------------------
    retrieval = box(ax, 0.95, 8.35, 1.7, 0.95, "Literature\nretrieval", "agent")
    hypothesis = box(ax, 2.95, 8.35, 1.7, 0.95, "Hypothesis\nagent", "agent")
    debate = box(ax, 4.95, 8.35, 1.7, 0.95, "Debate\nagents", "agent")
    codegen = box(ax, 6.95, 8.35, 1.7, 0.95, "Code-gen\nagent", "agent")
    for a, b in [(retrieval, hypothesis), (hypothesis, debate), (debate, codegen)]:
        arrow(ax, anchor(a, "E"), anchor(b, "W"), "flow")

    # ---- Entry path II: evolution ----------------------------------------
    parents = box(ax, 10.55, 9.0, 2.35, 0.95,
                  "Parent programs\n+ reflection briefs", "state")
    variate = box(ax, 10.55, 7.65, 2.35, 0.95, "LLM mutation\n/ crossover", "agent")
    jitter = box(ax, 13.15, 7.65, 2.2, 0.95, "Window-jitter\n(deterministic)", "det")
    arrow(ax, anchor(parents, "S"), anchor(variate, "N"), "flow")
    # parents -> window-jitter (up, across, and down into its top)
    elbow(ax, [anchor(parents, "E"), (13.15, 9.0), anchor(jitter, "N")], "flow")

    # ---- Deterministic core (spine) --------------------------------------
    compile_ = box(ax, 6.9, 5.55, 4.0, 0.95, "In-memory compile & validation", "det")
    evaluator = box(ax, 6.9, 3.85, 6.0, 1.15,
                    "Deterministic evaluator\nscore candidate vs current book "
                    "on the\ngeneration's data window", "eval", weight="bold")
    selection = box(ax, 6.9, 2.15, 6.0, 1.0,
                    "Constrained NSGA-II selection\n"
                    "islands · Pareto non-domination · archive", "det")
    book = box(ax, 6.9, 0.55, 6.0, 1.0,
               "Population = accepted book\n"
               "($N_\\mathrm{trials}$ updated each scored candidate)", "state")

    arrow(ax, anchor(compile_, "S"), anchor(evaluator, "N"), "flow")
    arrow(ax, anchor(evaluator, "S"), anchor(selection, "N"), "flow")
    arrow(ax, anchor(selection, "S"), anchor(book, "N"), "flow")

    # both entry paths feed the compiler
    elbow(ax, [anchor(codegen, "S"), (6.9, 6.55), anchor(compile_, "N")], "flow")
    elabel(ax, 7.75, 6.55, "new program")
    # evolution feed leaves the mutation box slightly left of centre, enters
    # the compiler from the east (down, then left)
    elbow(ax, [(10.2, anchor(variate, "S")[1]), (10.2, 5.55),
               anchor(compile_, "E")], "flow")

    # ---- feedback channels ------------------------------------------------
    # (1) selection channel: the book supplies next generation's parents.
    #     Clean vertical corridor in the gap between the variate/parents
    #     boxes (east x=11.725) and the jitter box (west x=12.05).
    elbow(ax, [anchor(book, "E"), (11.9, 0.55), (11.9, 8.75),
               (anchor(parents, "E")[0], 8.75)], "data")
    elabel(ax, 12.75, 4.3, "parents drawn\neach generation", color=GOLD)

    # (2) selection channel: candidate scored *against* the current book.
    elbow(ax, [anchor(book, "W"), (2.7, 0.55), (2.7, 3.85), anchor(evaluator, "W")],
          "data")
    elabel(ax, 2.7, 2.2, "current\nbook", color=GOLD)

    # (3) teacher channel: rendered diagnostics -> proposing agents. Enters
    #     the mutation box from below, right of the evolution-feed line.
    elbow(ax, [anchor(evaluator, "E"), (10.9, 3.85),
               (10.9, anchor(variate, "S")[1])], "teacher")
    elabel(ax, 11.35, 6.2, "reflection\nbriefs", color=AGENT, bg="#EEF3FB")

    # ---- group frames + agent boundary -----------------------------------
    AGENT_FILL_LIGHT = "#F3F6FC"
    group_frame(ax, 0.1, 7.88, 7.8, 8.82, AGENT, AGENT_FILL_LIGHT)
    group_frame(ax, 9.28, 7.18, 14.25, 9.47, AGENT, AGENT_FILL_LIGHT)
    # boundary encloses the two proposer groups only; kept inside the axes so
    # both vertical borders render, with headroom above the group titles.
    group_frame(ax, 0.1, 7.18, 14.25, 9.72, AGENT, None, pad=0.28,
                ls=(0, (5, 3)), lw=1.2)

    ax.text(0.1, 8.98, "Ideation  (literature-grounded)", fontsize=9,
            weight="bold", color=AGENT, ha="left", va="bottom")
    ax.text(9.28, 9.58, "Evolution  (self-improvement)", fontsize=9,
            weight="bold", color=AGENT, ha="left", va="bottom")
    ax.text(14.45, 9.93, "Agent boundary (LLM)", fontsize=8.5, style="italic",
            color=AGENT, ha="right", va="top")

    # ---- legend: compact two-row horizontal band -------------------------
    _horizontal_legend(ax, y_top=-0.95)

    fig.tight_layout(pad=0.3)
    return fig


def _horizontal_legend(ax, y_top):
    """Compact report-style legend: roles on row 1, channels on row 2."""
    x0, x1 = -0.35, 13.55
    y1, y2 = y_top - 0.42, y_top - 1.02
    group_frame(ax, x0, y_top - 1.35, x1, y_top, "#8A909C", "white",
                pad=0.12, lw=0.9)

    def swatch(x, y, role):
        fc, ec, lw = ROLE[role]
        ax.add_patch(FancyBboxPatch((x, y - 0.12), 0.44, 0.24,
                                    boxstyle="round,pad=0.005,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=lw, zorder=4))

    def line(x, y, style):
        cfg = {"flow": (INK, "-"), "data": (GOLD, (0, (3.0, 1.7))),
               "teacher": (AGENT, (0, (4.5, 2.2)))}
        col, ls = cfg[style]
        ax.plot([x, x + 0.5], [y, y], color=col, lw=1.6, ls=ls,
                solid_capstyle="round", zorder=4)

    def label(x, y, text):
        ax.text(x, y, text, fontsize=8.0, color=INK, va="center", ha="left")

    ax.text(x0 + 0.18, y_top - 0.02, "Legend", fontsize=8.5, weight="bold",
            color=INK, ha="left", va="top")

    # row 1 — node roles
    roles = [(0.30, "agent", "LLM agent"),
             (3.05, "det", "deterministic mechanism"),
             (6.75, "state", "state / accepted book"),
             (10.15, "eval", "deterministic evaluator")]
    for x, role, text in roles:
        swatch(x, y1, role)
        label(x + 0.60, y1, text)

    # row 2 — channels
    chans = [(0.30, "flow", "structural flow"),
             (3.05, "data", "selection channel (market numbers)"),
             (8.05, "teacher", "teacher channel (rendered diagnostics)")]
    for x, style, text in chans:
        line(x, y2, style)
        label(x + 0.62, y2, text)

    return ax


def main():
    here = Path(__file__).resolve().parent
    fig = build()
    for ext in ("pdf", "png"):
        out = here / f"fig_research_loop.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
