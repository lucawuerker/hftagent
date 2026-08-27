"""Deep-dive figures for arms 4 (L1H_terra_s0b) and 6 (L4WF_terra_s0) vs the
101 formulaic alphas, for the thesis results chapter.

All inputs are existing artifacts — no signal is recomputed:
  * factor signals from the shared parquet store
    (data/comparisons/wf_arm_analysis/signal_store/, 100% cached for all
    three books); correlations follow the wf_arm_factor_analysis convention
    (fit window < 2021-07-20, ~400 strided bars, signals flattened across
    tickers, NaN->0, Pearson);
  * per-factor fit/forward-block ICs from wf_arm_analysis_local/<arm>/
    per_factor_blocks.csv (arm 6 from thesis_ablation/tables/per_factor_all.csv,
    same convention/script);
  * prequential curves from the runs' prequential records; the zoo has no run,
    its analogue is the PIT LightGBM refit (flagged in the caption);
  * PIT Lasso curves from the pit_combiners races (arm 4 book =
    L1HCUR_terra_s0b keep-fids race; arm 6 from thesis_ablation pit_methods.csv;
    zoo from zoo.jsonl).

Output: data/comparisons/thesis_final_figures/*.png (300 dpi, no titles).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
ANA = REPO / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
TAB = REPO / "data/comparisons/thesis_ablation/tables"
STORE = REPO / "data/comparisons/wf_arm_analysis/signal_store"
OUT = REPO / "data/comparisons/thesis_final_figures"

WF_START = pd.Timestamp("2021-07-20")

BOOKS = {  # label -> (source arm, color, short name used in legends)
    "L1H": ("L1H_terra_s0b", "#2a78d6", "arm 4 (L1H)"),
    "L4WF": ("L4WF_terra_s0", "#eb6834", "arm 6 (L4WF)"),
    "zoo": ("zoo", "#1baf7a", "101 alphas"),
}
INK = "#1a1a18"
INK_MUTED = "#6e6d68"
GRID = "#e4e2dd"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "corr", ["#c94040", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "text.color": INK, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------- signals
def load_book(arm: str) -> dict[str, str]:
    if arm == "zoo":
        pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
        return {m["factor_id"]: m["code"] for m in pb["members"]}
    db = json.loads((WS / arm / "factors/factor_db.json").read_text())
    out = {}
    for r in db["factors"]:
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        out[r["id"]] = path.read_text()
    return out


def signal_matrix() -> tuple[np.ndarray, list[tuple[str, str]]]:
    """(n_obs x n_factors) flattened fit-window signal matrix + (book, fid)."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from wf_common import signal_key

    cols_meta, vecs = [], []
    ref_index = ref_cols = None
    for book, (arm, _c, _n) in BOOKS.items():
        for fid, code in load_book(arm).items():
            sig = pd.read_parquet(STORE / f"{signal_key(fid, code)}.parquet")
            if ref_index is None:
                fit_idx = sig.index[sig.index < WF_START]
                stride = max(1, len(fit_idx) // 400)
                ref_index = fit_idx[::stride]
                ref_cols = sig.columns
            v = sig.reindex(index=ref_index, columns=ref_cols) \
                   .to_numpy(dtype=float).ravel()
            vecs.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            cols_meta.append((book, fid))
    return np.column_stack(vecs), cols_meta


def cluster_order(c: np.ndarray) -> np.ndarray:
    d = squareform(np.clip(1.0 - np.abs(c), 0.0, None), checks=False)
    link = hierarchy.linkage(d, method="average", optimal_ordering=True)
    return np.asarray(hierarchy.leaves_list(link))


# ---------------------------------------------------------------- corr figures
def corr_heatmap(c: np.ndarray, fname: str, seps: list[int] | None = None,
                 sep_labels: list[str] | None = None, size: float = 5.2) -> None:
    fig, ax = plt.subplots(figsize=(size + 1.1, size))
    im = ax.imshow(c, cmap=DIV_CMAP, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if seps:
        for s in seps[:-1]:
            ax.axhline(s - 0.5, color="white", lw=1.6)
            ax.axvline(s - 0.5, color="white", lw=1.6)
        if sep_labels:
            bounds = [0] + seps
            for lo, hi, lab in zip(bounds[:-1], bounds[1:], sep_labels):
                mid = (lo + hi - 1) / 2
                ax.text(-0.02, mid, lab, transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=11, rotation=90)
                ax.text(mid, 1.01, lab, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label(r"pairwise signal correlation $\rho$")
    cb.outline.set_visible(False)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- curves
def _pit_curve(jsonl: Path, label: str, method: str) -> pd.Series:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines()]
    rows = [r for r in rows if r["label"] == label and r["method"] == method]
    return pd.Series({int(r["block_gen"]) - 10: r["ic"] for r in rows}).sort_index()


def curve_chart(series: dict[str, tuple[pd.Series, str, str]], fname: str,
                ylabel: str, block_starts: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for name, (s, color, style) in series.items():
        ax.plot(s.index, s.values, style, color=color, label=name,
                lw=2.0, ms=6, zorder=3)
    ax.axhline(0, color=INK, lw=0.9, zorder=2)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"{b}\n{d}" for b, d in
                        zip(range(1, 11), block_starts)], fontsize=8.5)
    ax.set_xlabel("walk-forward block (start date)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- scatter
def per_factor_frames() -> dict[str, pd.DataFrame]:
    out = {}
    for book, arm in (("L1H", "L1H_terra_s0b"), ("zoo", "zoo")):
        d = pd.read_csv(ANA / arm / "per_factor_blocks.csv")
        out[book] = d.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    pf = pd.read_csv(TAB / "per_factor_all.csv")
    out["L4WF"] = pf[pf["arm"] == "6"].dropna(
        subset=["ic_fit", "ic_wf_blockmean"])
    return out


def scatter_fit_vs_wf(frames: dict[str, pd.DataFrame], fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lim = 0.0
    for book, (arm, color, name) in BOOKS.items():
        d = frames[book]
        ax.scatter(d["ic_fit"], d["ic_wf_blockmean"], s=26, color=color,
                   alpha=0.75, edgecolors="white", linewidths=0.5,
                   label=name, zorder=3)
        lim = max(lim, d["ic_fit"].abs().max(), d["ic_wf_blockmean"].abs().max())
    lim *= 1.08
    ax.plot([-lim, lim], [-lim, lim], ls="--", color=INK_MUTED, lw=1.0,
            zorder=2)
    ax.axhline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=INK_MUTED, lw=0.8, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("per-factor IC, fit window (to 2021-07)")
    ax.set_ylabel("per-factor mean IC, 10 forward blocks")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- max-corr
def _plain(name: str) -> str:
    """Legend name without the run label, e.g. "arm 4 (L1H)" -> "arm 4"."""
    return name.split(" (")[0]


def grouped_hist(data: dict[str, tuple[np.ndarray, str]], fname: str,
                 xlabel: str, bins: np.ndarray) -> None:
    n = len(data)
    width = (bins[1] - bins[0]) / (n + 0.6)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (name, (vals, color)) in enumerate(data.items()):
        counts, _ = np.histogram(vals, bins=bins)
        share = counts / counts.sum()
        ax.bar(bins[:-1] + (i + 0.5) * width, share, width=width * 0.92,
               color=color, label=name, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("share of factors")
    ax.set_xticks(bins[::2])
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mat, meta = signal_matrix()
    corr = np.corrcoef(mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)  # zero-variance columns -> rho 0
    np.fill_diagonal(corr, 1.0)
    books = np.array([b for b, _ in meta])
    print("signal matrix", mat.shape, "corr", corr.shape)

    # within-book matrices (clustered) + combined
    order_all = []
    seps, running = [], 0
    for book in BOOKS:
        idx = np.flatnonzero(books == book)
        sub = corr[np.ix_(idx, idx)]
        o = idx[cluster_order(sub)]
        corr_heatmap(corr[np.ix_(o, o)], f"corr_book_{book.lower()}.png",
                     size=4.6)
        order_all.extend(o.tolist())
        running += len(idx)
        seps.append(running)
    o = np.array(order_all)
    corr_heatmap(corr[np.ix_(o, o)], "corr_all_books.png", seps=seps,
                 sep_labels=["L1H", "L4WF", "101 alphas"], size=6.4)

    # summary stats for the caption / text
    stats = {}
    for book in BOOKS:
        idx = np.flatnonzero(books == book)
        sub = np.abs(corr[np.ix_(idx, idx)])
        np.fill_diagonal(sub, np.nan)
        stats[book] = {
            "mean_abs": float(np.nanmean(sub)),
            "median_max_within": float(np.median(np.nanmax(sub, axis=1))),
        }
    li, l4, zi = (np.flatnonzero(books == b) for b in ("L1H", "L4WF", "zoo"))
    cross = np.abs(corr[np.ix_(li, l4)])
    stats["cross_L1H_L4WF_mean_abs"] = float(cross.mean())
    print(json.dumps(stats, indent=2))

    # prequential + lasso curves over the 10 blocks
    p4 = pd.read_csv(ANA / "L1H_terra_s0b/prequential_record.csv")
    p4 = p4[p4["generation"] >= 11]
    s4 = pd.Series(p4["combined_oos_ic"].values,
                   index=p4["generation"].values - 10)
    block_starts = [s[:7] for s in p4["start"].str[:10]]
    pq = pd.read_csv(TAB / "prequential_blocks.csv")
    p6 = pq[pq["run"] == "L4WF_terra_s0"].sort_values("block")
    s6 = pd.Series(p6["ic"].values, index=p6["block"].values)
    zoo_gbm = _pit_curve(PIT / "zoo.jsonl", "zoo", "lightgbm")
    curve_chart({
        "arm 4": (s4, BOOKS["L1H"][1], "-o"),
        "arm 6": (s6, BOOKS["L4WF"][1], "-o"),
        "101 alphas": (zoo_gbm, BOOKS["zoo"][1], "--s"),
    }, "curves_prequential.png", "prequential block IC", block_starts)

    l4_lasso = pd.read_csv(TAB / "pit_methods.csv")
    l4_lasso = l4_lasso[(l4_lasso["run"] == "L4WF_terra_s0")
                        & (l4_lasso["book"] == "book")
                        & (l4_lasso["method"] == "lasso")].sort_values("block")
    s6l = pd.Series(l4_lasso["ic"].values, index=l4_lasso["block"].values)
    s4l = _pit_curve(PIT / "L1HCUR_terra_s0b.jsonl", "L1HCUR_terra_s0b", "lasso")
    szl = _pit_curve(PIT / "zoo.jsonl", "zoo", "lasso")
    curve_chart({
        "arm 4": (s4l, BOOKS["L1H"][1], "-o"),
        "arm 6": (s6l, BOOKS["L4WF"][1], "-o"),
        "101 alphas": (szl, BOOKS["zoo"][1], "--s"),
    }, "curves_lasso.png", "PIT Lasso block IC (curated book)", block_starts)

    # fit vs forward scatter
    scatter_fit_vs_wf(per_factor_frames(), "scatter_fit_vs_wf.png")

    # max-|rho| analyses
    bins = np.arange(0.0, 1.05, 0.1)
    within = {}
    for book, (_a, color, name) in BOOKS.items():
        idx = np.flatnonzero(books == book)
        sub = np.abs(corr[np.ix_(idx, idx)])
        np.fill_diagonal(sub, np.nan)
        within[_plain(name)] = (np.nanmax(sub, axis=1), color)
    grouped_hist(within, "maxcorr_within_book.png",
                 r"max $|\rho|$ to another member of the same book", bins)

    vs_zoo = {}
    for book in ("L1H", "L4WF"):
        _a, color, name = BOOKS[book]
        idx = np.flatnonzero(books == book)
        cross = np.abs(corr[np.ix_(idx, zi)])
        vs_zoo[_plain(name)] = (cross.max(axis=1), color)
    grouped_hist(vs_zoo, "maxcorr_vs_zoo.png",
                 r"max $|\rho|$ to any of the 101 formulaic alphas", bins)

    (OUT / "deepdive_stats.json").write_text(json.dumps(stats, indent=2))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
