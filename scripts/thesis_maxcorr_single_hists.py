"""Split Figure 4.2 into individual per-book histograms.

Supervisor request (p64): panel (a) as two separate histograms (arm 4, arm 6)
and panel (b) as three (arm 4, arm 6, the 101 formulaic alphas), because the
grouped bars are hard to read at print size.

Everything is recomputed from the SAME inputs and with the SAME convention as
scripts/thesis_book_deepdive_figures.py — that module is imported and its
signal matrix, correlation convention, bins, colours and rcParams are reused,
so the split panels are guaranteed to show the identical numbers as the
grouped originals (maxcorr_within_book.png, maxcorr_vs_zoo.png).

Output: data/comparisons/thesis_final_figures/maxcorr_{within,vs_zoo}_<book>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import thesis_book_deepdive_figures as dd

BINS = np.arange(0.0, 1.05, 0.1)
SLUG = {"arm 4 (L1H)": "arm4", "arm 6 (L4WF)": "arm6", "101 alphas": "zoo"}


def single_hist(vals: np.ndarray, color: str, label: str,
                fname: str, xlabel: str, ymax: float) -> None:
    """One book's distribution, same bins/colour/normalisation as the group."""
    counts, _ = np.histogram(vals, bins=BINS)
    share = counts / counts.sum()
    width = BINS[1] - BINS[0]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(BINS[:-1] + width / 2, share, width=width * 0.88,
           color=color, label=label, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("share of factors")
    ax.set_xticks(BINS[::2])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, ymax)          # shared across the panels of one figure
    ax.grid(axis="y", color=dd.GRID, lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(dd.OUT / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}: n={len(vals)} median={np.median(vals):.3f} "
          f"mean={vals.mean():.3f}")


def main() -> None:
    dd.OUT.mkdir(parents=True, exist_ok=True)
    mat, meta = dd.signal_matrix()
    corr = np.nan_to_num(np.corrcoef(mat, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 1.0)
    books = np.array([b for b, _ in meta])
    print("signal matrix", mat.shape)

    # ── panel (b): maximum |rho| to another member of the SAME book ────────
    within = {}
    for key, (_arm, color, name) in dd.BOOKS.items():
        idx = np.flatnonzero(books == key)
        sub = np.abs(corr[np.ix_(idx, idx)])
        np.fill_diagonal(sub, np.nan)
        within[name] = (np.nanmax(sub, axis=1), color)
    ymax = max(np.histogram(v, bins=BINS)[0].max() / len(v)
               for v, _ in within.values()) * 1.08
    xlab = r"max $|\rho|$ to another member of the same book"
    print("within-book:")
    for name, (vals, color) in within.items():
        single_hist(vals, color, dd._plain(name),
                    f"maxcorr_within_{SLUG[name]}.png", xlab, ymax)

    # ── panel (a): maximum |rho| to any of the 101 formulaic alphas ────────
    zi = np.flatnonzero(books == "zoo")
    vs_zoo = {}
    for key in ("L1H", "L4WF"):
        _arm, color, name = dd.BOOKS[key]
        idx = np.flatnonzero(books == key)
        vs_zoo[name] = (np.abs(corr[np.ix_(idx, zi)]).max(axis=1), color)
    ymax = max(np.histogram(v, bins=BINS)[0].max() / len(v)
               for v, _ in vs_zoo.values()) * 1.08
    xlab = r"max $|\rho|$ to any of the 101 formulaic alphas"
    print("vs the 101 alphas:")
    for name, (vals, color) in vs_zoo.items():
        single_hist(vals, color, dd._plain(name),
                    f"maxcorr_vs_zoo_{SLUG[name]}.png", xlab, ymax)

    print("done ->", dd.OUT)


if __name__ == "__main__":
    main()
