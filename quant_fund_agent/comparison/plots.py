"""Presentation-ready figures for the research-LLM comparison (matplotlib only).

Every function takes a tidy results DataFrame and writes one PNG into
``figures_dir``, returning the path (or ``None`` if there's nothing to plot).
Styling is consistent and thesis-ready: a fixed prerun colour map, value labels,
zero reference lines, tight layout, dpi 150.  No seaborn dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("comparison.plots")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    "font.size": 10, "figure.autolayout": False,
})
_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04"]


def _colour_map(keys: list[str]) -> dict[str, str]:
    return {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(sorted(keys))}


def _save(fig, figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote figure %s", path)
    return path


def _nums(seq) -> list[float]:
    """Coerce a sequence to floats, mapping None / non-numeric → NaN.

    matplotlib's ``bar`` rejects ``None`` heights (``int + None``); NaN is fine
    (the bar is simply skipped).  Degenerate runs (e.g. a 1-ticker universe with
    no cross-section → all-None IC) must not crash figure rendering.
    """
    out: list[float] = []
    for v in seq:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def _label_bars(ax, bars, fmt="{:.2f}", fontsize=7) -> None:
    for b in bars:
        h = b.get_height()
        if h is None or (isinstance(h, float) and np.isnan(h)):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=fontsize, xytext=(0, 1 if h >= 0 else -1),
                    textcoords="offset points")


# ── Track 1: single-factor IC ───────────────────────────────────────────────

def ic_distribution(ic_rows: list[dict[str, Any]], figures_dir: Path, h: int = 6) -> Path | None:
    """Box plot of per-factor |IC| at horizon ``h``, one box per prerun."""
    df = pd.DataFrame(ic_rows)
    col = f"ic_{h}"
    if df.empty or col not in df:
        return None
    preruns = sorted(df["prerun"].unique())
    data = [pd.to_numeric(df.loc[df.prerun == p, col], errors="coerce").abs().dropna().values
            for p in preruns]
    if not any(len(d) for d in data):
        return None
    cmap = _colour_map(preruns)
    fig, ax = plt.subplots(figsize=(1.6 * len(preruns) + 3, 4.2))
    bp = ax.boxplot(data, tick_labels=preruns, showmeans=True, patch_artist=True)
    for patch, p in zip(bp["boxes"], preruns):
        patch.set_facecolor(cmap[p])
        patch.set_alpha(0.45)
    ax.set_ylabel(f"|IC| at h={h} ({h * 10}s)")
    ax.set_title(f"Single-factor |IC| distribution by research model (h={h})")
    ax.axhline(0.02, color="grey", ls="--", lw=0.8, label="|IC|=0.02 ref")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, f"ic_distribution_h{h}.png")


def ic_mean_by_horizon(ic_summary: list[dict[str, Any]], figures_dir: Path,
                       horizons: tuple[int, ...] = (1, 6, 60)) -> Path | None:
    """Grouped bars: mean |IC| per prerun across horizons."""
    df = pd.DataFrame(ic_summary)
    if df.empty:
        return None
    preruns = list(df["prerun"])
    cmap = _colour_map(preruns)
    x = np.arange(len(horizons))
    width = 0.8 / max(1, len(preruns))
    fig, ax = plt.subplots(figsize=(2 * len(horizons) + 3, 4.2))
    drew = False
    for i, p in enumerate(preruns):
        row = df[df.prerun == p].iloc[0]
        vals = _nums([row.get(f"mean_abs_ic_{h}") for h in horizons])
        drew = drew or any(not np.isnan(v) for v in vals)
        bars = ax.bar(x + i * width, vals, width, label=p, color=cmap[p], alpha=0.85)
        _label_bars(ax, bars, fmt="{:.3f}")
    if not drew:
        plt.close(fig)
        return None
    ax.set_xticks(x + width * (len(preruns) - 1) / 2)
    ax.set_xticklabels([f"h={h}\n({h * 10}s)" for h in horizons])
    ax.set_ylabel("mean |IC| across factors")
    ax.set_title("Mean single-factor |IC| by research model and horizon")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "ic_mean_by_horizon.png")


def ic_top_factors(ic_rows: list[dict[str, Any]], figures_dir: Path,
                   h: int = 6, top_n: int = 10) -> Path | None:
    """Per-prerun horizontal bars of the top-N factors by |IC| at horizon ``h``."""
    df = pd.DataFrame(ic_rows)
    col = f"ic_{h}"
    if df.empty or col not in df:
        return None
    preruns = sorted(df["prerun"].unique())
    cmap = _colour_map(preruns)
    fig, axes = plt.subplots(1, len(preruns), figsize=(3.2 * len(preruns) + 1, 4.4),
                             squeeze=False)
    for ax, p in zip(axes[0], preruns):
        g = df[df.prerun == p].copy()
        g["abs_ic"] = pd.to_numeric(g[col], errors="coerce").abs()
        g = g.dropna(subset=["abs_ic"]).sort_values("abs_ic", ascending=True).tail(top_n)
        ax.barh(g["factor_id"].astype(str).str.slice(0, 22), g["abs_ic"],
                color=cmap[p], alpha=0.85)
        ax.set_title(p, fontsize=10)
        ax.set_xlabel(f"|IC| h={h}")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Top {top_n} factors by |IC| (h={h}) per research model", y=1.02)
    return _save(fig, figures_dir, f"ic_top_factors_h{h}.png")


# ── Track 2: brute-force ML ──────────────────────────────────────────────────

def bruteforce_oos_sharpe(bf_rows: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Grouped bars of OOS Sharpe by model, one colour per prerun."""
    df = pd.DataFrame(bf_rows)
    if df.empty or "oos_sharpe" not in df:
        return None
    df = df.dropna(subset=["oos_sharpe"])
    if df.empty:
        return None
    preruns = sorted(df["prerun"].unique())
    models = list(dict.fromkeys(df["model"]))  # preserve catalog order
    cmap = _colour_map(preruns)
    x = np.arange(len(models))
    width = 0.8 / max(1, len(preruns))
    fig, ax = plt.subplots(figsize=(1.3 * len(models) + 3, 4.4))
    for i, p in enumerate(preruns):
        sub = df[df.prerun == p].set_index("model")
        vals = [sub.loc[m, "oos_sharpe"] if m in sub.index else np.nan for m in models]
        bars = ax.bar(x + i * width, vals, width, label=p, color=cmap[p], alpha=0.85)
        _label_bars(ax, bars)
    ax.set_xticks(x + width * (len(preruns) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("ML-combined signal: out-of-sample Sharpe by model and research set")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "bruteforce_oos_sharpe.png")


def bruteforce_ic_heatmap(bf_rows: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Heatmap of OOS IC over (prerun × model)."""
    df = pd.DataFrame(bf_rows)
    if df.empty or "oos_ic" not in df:
        return None
    piv = df.pivot_table(index="prerun", columns="model", values="oos_ic", aggfunc="first")
    if piv.empty:
        return None
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 2, 0.7 * piv.shape[0] + 2))
    vmax = float(np.nanmax(np.abs(piv.values))) or 1e-6
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="OOS IC", fraction=0.046, pad=0.04)
    ax.set_title("ML-combined signal: out-of-sample IC (prerun × model)")
    return _save(fig, figures_dir, "bruteforce_ic_heatmap.png")


def bruteforce_is_vs_oos(bf_rows: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """IS-vs-OOS Sharpe scatter — the overfitting diagnostic."""
    df = pd.DataFrame(bf_rows)
    if df.empty or "is_sharpe" not in df or "oos_sharpe" not in df:
        return None
    df = df.dropna(subset=["is_sharpe", "oos_sharpe"])
    if df.empty:
        return None
    preruns = sorted(df["prerun"].unique())
    cmap = _colour_map(preruns)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p in preruns:
        sub = df[df.prerun == p]
        ax.scatter(sub["is_sharpe"], sub["oos_sharpe"], color=cmap[p], label=p, s=45, alpha=0.8)
        for _, r in sub.iterrows():
            ax.annotate(str(r["model"])[:10], (r["is_sharpe"], r["oos_sharpe"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    lo = float(np.nanmin(df[["is_sharpe", "oos_sharpe"]].values))
    hi = float(np.nanmax(df[["is_sharpe", "oos_sharpe"]].values))
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=0.8, label="IS = OOS")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("IS Sharpe")
    ax.set_ylabel("OOS Sharpe")
    ax.set_title("ML-combined signal: in-sample vs out-of-sample Sharpe")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "bruteforce_is_vs_oos.png")


# ── Track A: diversity & redundancy ──────────────────────────────────────────

def factor_correlation_heatmap(corr, figures_dir: Path, prerun: str) -> Path | None:
    """Heatmap of the pairwise signal correlation matrix for one prerun."""
    if corr is None or getattr(corr, "empty", True) or corr.shape[0] < 2:
        return None
    C = corr.to_numpy(dtype=float)
    n = C.shape[0]
    fig, ax = plt.subplots(figsize=(0.32 * n + 2.5, 0.32 * n + 2))
    im = ax.imshow(np.nan_to_num(C, nan=0.0), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    if n <= 30:
        labels = [str(c)[:16] for c in corr.columns]
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="signal corr", fraction=0.046, pad=0.04)
    ax.set_title(f"Factor signal correlation — {prerun} (n={n})")
    return _save(fig, figures_dir, f"factor_correlation_{prerun}.png")


def effective_factors_bar(div_summary: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Per-prerun bars: effective # independent factors vs the raw count."""
    df = pd.DataFrame(div_summary)
    if df.empty or "eff_n_factors" not in df:
        return None
    preruns = list(df["prerun"])
    cmap = _colour_map(preruns)
    x = np.arange(len(preruns))
    fig, ax = plt.subplots(figsize=(1.8 * len(preruns) + 3, 4.4))
    b1 = ax.bar(x - 0.2, _nums(df["n_factors"]), 0.4, label="# factors",
                color="#9ca3af", alpha=0.7)
    b2 = ax.bar(x + 0.2, _nums(df["eff_n_factors"]), 0.4, label="effective # (independent)",
                color=[cmap[p] for p in preruns], alpha=0.9)
    _label_bars(ax, b1, fmt="{:.0f}"); _label_bars(ax, b2, fmt="{:.1f}")
    ax.set_xticks(x); ax.set_xticklabels(preruns, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("# factors")
    ax.set_title("Factor zoo diversity: effective vs raw factor count")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "effective_factors.png")


# ── Track B: deflation & model-based importance ──────────────────────────────

def deflation_bar(mv_summary: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Per-prerun bars: best |IC| vs the selection-deflated best |IC|."""
    df = pd.DataFrame(mv_summary)
    if df.empty or "best_ic" not in df or not df["best_ic"].notna().any():
        return None
    preruns = list(df["prerun"])
    cmap = _colour_map(preruns)
    x = np.arange(len(preruns))
    fig, ax = plt.subplots(figsize=(1.8 * len(preruns) + 3, 4.4))
    b1 = ax.bar(x - 0.2, _nums(df["best_ic"]), 0.4, label="best |IC|",
                color="#9ca3af", alpha=0.7)
    b2 = ax.bar(x + 0.2, _nums(df.get("deflated_best_ic", [np.nan] * len(preruns))), 0.4,
                label="deflated best |IC|", color=[cmap[p] for p in preruns], alpha=0.9)
    _label_bars(ax, b1, fmt="{:.3f}"); _label_bars(ax, b2, fmt="{:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(preruns, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("|IC| at default horizon")
    ax.set_title("Best single-factor |IC| before vs after multiple-testing deflation")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "deflation.png")


def feature_importance_bar(importance_rows: list[dict[str, Any]], figures_dir: Path,
                           model: str | None = None, top_n: int = 10) -> Path | None:
    """Per-prerun horizontal bars of the top factors by model importance."""
    df = pd.DataFrame(importance_rows)
    if df.empty or "importance" not in df:
        return None
    if model is None:
        model = df["model"].iloc[0]
    df = df[df["model"] == model]
    if df.empty:
        return None
    preruns = sorted(df["prerun"].unique())
    cmap = _colour_map(preruns)
    fig, axes = plt.subplots(1, len(preruns), figsize=(3.2 * len(preruns) + 1, 4.4),
                             squeeze=False)
    for ax, p in zip(axes[0], preruns):
        g = df[df.prerun == p].sort_values("importance", ascending=True).tail(top_n)
        ax.barh(g["factor_id"].astype(str).str.slice(0, 22), g["importance"],
                color=cmap[p], alpha=0.85)
        ax.set_title(p, fontsize=10)
        ax.set_xlabel(f"{model} importance")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Top {top_n} factors by {model} importance per research model", y=1.02)
    return _save(fig, figures_dir, f"feature_importance_{model}.png")


# ── Track 3: downstream agents ───────────────────────────────────────────────

def downstream_summary(ds_rows: list[dict[str, Any]], figures_dir: Path) -> Path | None:
    """Twin bars: per-prerun mean OOS Sharpe of approved strategies + #approved."""
    df = pd.DataFrame(ds_rows)
    if df.empty:
        return None
    preruns = list(df["prerun"])
    cmap = _colour_map(preruns)
    x = np.arange(len(preruns))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * len(preruns) + 4, 4.2))
    sharpe = _nums([df[df.prerun == p].iloc[0].get("mean_oos_sharpe") for p in preruns])
    b1 = ax1.bar(x, sharpe, color=[cmap[p] for p in preruns], alpha=0.85)
    _label_bars(ax1, b1)
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(preruns, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("mean OOS Sharpe (approved)")
    ax1.set_title("Downstream: approved-strategy OOS Sharpe")
    appr = [df[df.prerun == p].iloc[0].get("n_approved", 0) for p in preruns]
    att = [df[df.prerun == p].iloc[0].get("n_attempted", 0) for p in preruns]
    b2 = ax2.bar(x, appr, color=[cmap[p] for p in preruns], alpha=0.85)
    _label_bars(ax2, b2, fmt="{:.0f}")
    for xi, a in zip(x, att):
        ax2.annotate(f"/{a}", (xi, 0), ha="center", va="bottom", fontsize=7, color="grey")
    ax2.set_xticks(x); ax2.set_xticklabels(preruns, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("# strategies approved")
    ax2.set_title("Downstream: strategies approved / attempted")
    return _save(fig, figures_dir, "downstream_summary.png")


# ── shared: factor usability on the current data ─────────────────────────────

def factor_usability(usability: dict[str, dict[str, Any]], figures_dir: Path) -> Path | None:
    """Stacked bars: usable vs dropped factors per prerun (the current-data story)."""
    if not usability:
        return None
    preruns = sorted(usability)
    usable = [usability[p]["n_usable"] for p in preruns]
    dropped = [usability[p]["n_dropped"] for p in preruns]
    x = np.arange(len(preruns))
    fig, ax = plt.subplots(figsize=(1.6 * len(preruns) + 3, 4.2))
    b1 = ax.bar(x, usable, color="#16a34a", alpha=0.85, label="usable now")
    b2 = ax.bar(x, dropped, bottom=usable, color="#9ca3af", alpha=0.7,
                label="needs more data")
    _label_bars(ax, b1, fmt="{:.0f}")
    for xi, u, d in zip(x, usable, dropped):
        if d:
            ax.annotate(f"{d:.0f}", (xi, u + d), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(preruns, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("# researched factors")
    ax.set_title("Factor usability on the current data (usable vs awaiting data)")
    ax.legend(fontsize=8)
    return _save(fig, figures_dir, "factor_usability.png")
