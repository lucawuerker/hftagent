"""Build report-ready figures for a rolling factor-set comparison batch.

Usage:
    ./venv/bin/python scripts/plot_rolling_factor_report_figures.py \
        data/comparisons/rolling_20260624_001308

The script reads the aggregated CSVs under ``combined/`` and writes compact
figures to ``report_figures/`` inside the batch directory.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qfa-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "gpt4omini120650": "GPT-4o mini",
    "gpt5.4mini120650": "GPT-5.4 mini",
    "main": "Seed factors",
}

ORDER = ["gpt4omini120650", "gpt5.4mini120650", "main"]
MODEL_ORDER = [
    "linear_regression",
    "ridge",
    "lasso",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "ensemble",
]

PALETTE = {
    "gpt4omini120650": "#4C78A8",
    "gpt5.4mini120650": "#59A14F",
    "main": "#F28E2B",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
        }
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _label(x: str) -> str:
    return LABELS.get(x, x)


def _ordered_preruns(df: pd.DataFrame) -> list[str]:
    seen = set(df["prerun"].dropna())
    return [p for p in ORDER if p in seen] + sorted(seen - set(ORDER))


def _annotated_heatmap(
    values: pd.DataFrame,
    title: str,
    cbar_label: str,
    cmap: str = "RdYlGn",
    center: float | None = None,
    fmt: str = ".1f",
    figsize: tuple[float, float] = (7.2, 3.6),
) -> plt.Figure:
    arr = values.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize)

    if center is None:
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
    else:
        vmax = np.nanmax(np.abs(arr - center))
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=center - vmax, vmax=center + vmax)

    ax.set_title(title)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels(values.columns, rotation=35, ha="right")
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels(values.index)
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    return fig


def fig_ensemble_distribution(bf: pd.DataFrame, out_dir: Path) -> str:
    ens = bf.loc[bf["model"] == "ensemble"].copy()
    preruns = _ordered_preruns(ens)
    data = [ens.loc[ens["prerun"] == p, "oos_sharpe"].dropna().to_numpy() for p in preruns]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bp = ax.boxplot(
        data,
        tick_labels=[_label(p) for p in preruns],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, p in zip(bp["boxes"], preruns):
        patch.set_facecolor(PALETTE.get(p, "#777777"))
        patch.set_alpha(0.7)

    for i, p in enumerate(preruns, start=1):
        vals = ens.loc[ens["prerun"] == p, "oos_sharpe"].dropna()
        ax.scatter(i, vals.mean(), marker="D", s=24, color="#111111", zorder=3, label="Mean" if i == 1 else None)
        ax.text(i + 0.18, vals.quantile(0.95), f"p95={vals.quantile(0.95):.1f}", fontsize=7, va="center")

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("OOS Sharpe, ensemble")
    ax.set_title("Ensemble OOS Sharpe across rolling ETF-month windows")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig01_ensemble_oos_sharpe_distribution")
    return "fig01_ensemble_oos_sharpe_distribution"


def fig_model_heatmap(bf: pd.DataFrame, out_dir: Path) -> str:
    models = [m for m in MODEL_ORDER if m in set(bf["model"])]
    table = (
        bf.pivot_table(index="prerun", columns="model", values="oos_sharpe", aggfunc="median")
        .reindex(index=_ordered_preruns(bf), columns=models)
    )
    table.index = [_label(x) for x in table.index]
    table.columns = [m.replace("_", " ") for m in table.columns]
    fig = _annotated_heatmap(
        table,
        "Median OOS Sharpe by factor set and model",
        "Median OOS Sharpe",
        fmt=".1f",
        figsize=(8.0, 3.0),
    )
    _save(fig, out_dir, "fig02_model_robustness_median_oos_sharpe")
    return "fig02_model_robustness_median_oos_sharpe"


def fig_time_trend(bf: pd.DataFrame, out_dir: Path) -> str:
    ens = bf.loc[bf["model"] == "ensemble"].copy()
    ens["month"] = pd.to_datetime(ens["oos_month"] + "-01")
    grouped = ens.groupby(["prerun", "month"])["oos_sharpe"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for p in _ordered_preruns(ens):
        med = grouped.median().loc[p].sort_index()
        q25 = grouped.quantile(0.25).loc[p].sort_index()
        q75 = grouped.quantile(0.75).loc[p].sort_index()
        color = PALETTE.get(p, None)
        ax.plot(med.index, med.values, label=_label(p), color=color, linewidth=1.8)
        ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.15, linewidth=0)

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Median OOS Sharpe")
    ax.set_title("Time stability of ensemble OOS Sharpe")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.autofmt_xdate(rotation=30, ha="right")
    _save(fig, out_dir, "fig03_ensemble_oos_sharpe_over_time")
    return "fig03_ensemble_oos_sharpe_over_time"


def fig_ticker_heatmap(bf: pd.DataFrame, out_dir: Path) -> str:
    ens = bf.loc[bf["model"] == "ensemble"].copy()
    table = ens.pivot_table(index="ticker", columns="prerun", values="oos_sharpe", aggfunc="median")
    table = table.reindex(columns=_ordered_preruns(ens)).sort_index()
    table.columns = [_label(x) for x in table.columns]
    fig = _annotated_heatmap(
        table,
        "Median ensemble OOS Sharpe by ETF",
        "Median OOS Sharpe",
        fmt=".1f",
        figsize=(5.8, 5.0),
    )
    _save(fig, out_dir, "fig04_ticker_factor_set_heatmap")
    return "fig04_ticker_factor_set_heatmap"


def fig_ic_sharpe_scatter(bf: pd.DataFrame, out_dir: Path) -> str:
    ens = bf.loc[bf["model"] == "ensemble"].copy()
    # Keep the central range visible while retaining the full data for correlations.
    y_min, y_max = ens["oos_sharpe"].quantile([0.01, 0.99])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in _ordered_preruns(ens):
        sub = ens.loc[ens["prerun"] == p]
        ax.scatter(
            sub["oos_ic"],
            sub["oos_sharpe"].clip(y_min, y_max),
            s=15,
            alpha=0.55,
            label=_label(p),
            color=PALETTE.get(p, None),
            edgecolor="none",
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("OOS IC of combined signal")
    ax.set_ylabel("OOS Sharpe, clipped at 1st/99th pct.")
    ax.set_title("Signal IC versus realised ensemble Sharpe")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig05_oos_ic_vs_oos_sharpe")
    return "fig05_oos_ic_vs_oos_sharpe"


def fig_diversity(div: pd.DataFrame, out_dir: Path) -> str:
    preruns = _ordered_preruns(div)
    labels = [_label(p) for p in preruns]
    x = np.arange(len(preruns))
    width = 0.32

    means = div.groupby("prerun")[["eff_ratio", "redundancy", "mean_abs_corr", "n_factors"]].mean()
    q25 = div.groupby("prerun")[["eff_ratio", "redundancy"]].quantile(0.25)
    q75 = div.groupby("prerun")[["eff_ratio", "redundancy"]].quantile(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ax = axes[0]
    eff = means.loc[preruns, "eff_ratio"].to_numpy()
    red = means.loc[preruns, "redundancy"].to_numpy()
    eff_err = np.vstack([
        eff - q25.loc[preruns, "eff_ratio"].to_numpy(),
        q75.loc[preruns, "eff_ratio"].to_numpy() - eff,
    ])
    red_err = np.vstack([
        red - q25.loc[preruns, "redundancy"].to_numpy(),
        q75.loc[preruns, "redundancy"].to_numpy() - red,
    ])
    ax.bar(x - width / 2, eff, width, yerr=eff_err, label="Effective ratio", color="#4C78A8", capsize=2)
    ax.bar(x + width / 2, red, width, yerr=red_err, label="Redundancy", color="#E15759", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Diversity of factor libraries")
    ax.set_ylabel("Mean value across windows")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(
        means.loc[preruns, "mean_abs_corr"],
        means.loc[preruns, "eff_ratio"],
        s=90,
        color=[PALETTE.get(p, "#777777") for p in preruns],
    )
    for p in preruns:
        row = means.loc[p]
        ax.text(row["mean_abs_corr"] + 0.001, row["eff_ratio"], _label(p), va="center", fontsize=8)
    ax.set_xlabel("Mean absolute factor correlation")
    ax.set_ylabel("Effective factor ratio")
    ax.set_title("Correlation versus independence")
    fig.tight_layout()
    _save(fig, out_dir, "fig06_factor_library_diversity")
    return "fig06_factor_library_diversity"


def fig_importance_recurrence(imp: pd.DataFrame, out_dir: Path) -> str:
    # Count how often a factor appears among the top five importances across
    # ticker-months and importance models. This favors stable recurrence over one
    # very large coefficient in one window.
    top = imp.loc[imp["rank"] <= 5].copy()
    preruns = _ordered_preruns(top)
    fig, axes = plt.subplots(len(preruns), 1, figsize=(7.4, 6.2), sharex=False)
    if len(preruns) == 1:
        axes = [axes]

    for ax, p in zip(axes, preruns):
        sub = top.loc[top["prerun"] == p]
        counts = (
            sub.groupby(["factor_id", "name"])
            .size()
            .sort_values(ascending=False)
            .head(8)
            .sort_values()
        )
        labels = [name if isinstance(name, str) and name else fid for fid, name in counts.index]
        ax.barh(labels, counts.values, color=PALETTE.get(p, "#777777"), alpha=0.8)
        ax.set_title(f"{_label(p)}: recurring top-5 important factors")
        ax.set_xlabel("Top-5 appearances")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    fig.tight_layout()
    _save(fig, out_dir, "fig07_top_factor_importance_recurrence")
    return "fig07_top_factor_importance_recurrence"


def fig_quality_performance(div: pd.DataFrame, bf: pd.DataFrame, out_dir: Path) -> str:
    ens = bf.loc[bf["model"] == "ensemble", ["ticker", "oos_month", "prerun", "oos_sharpe"]]
    merged = div.merge(ens, on=["ticker", "oos_month", "prerun"], how="inner")
    y_min, y_max = merged["oos_sharpe"].quantile([0.01, 0.99])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for p in _ordered_preruns(merged):
        sub = merged.loc[merged["prerun"] == p]
        ax.scatter(
            sub["eff_ratio"],
            sub["oos_sharpe"].clip(y_min, y_max),
            s=16,
            alpha=0.5,
            label=_label(p),
            color=PALETTE.get(p, None),
            edgecolor="none",
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Effective factor ratio")
    ax.set_ylabel("Ensemble OOS Sharpe, clipped at 1st/99th pct.")
    ax.set_title("Factor-library independence versus OOS performance")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig08_independence_vs_oos_sharpe")
    return "fig08_independence_vs_oos_sharpe"


def write_summary_tables(bf: pd.DataFrame, div: pd.DataFrame, out_dir: Path) -> None:
    ens = bf.loc[bf["model"] == "ensemble"].copy()
    summary = ens.groupby("prerun")["oos_sharpe"].agg(
        count="count",
        mean="mean",
        median="median",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p95=lambda s: s.quantile(0.95),
    )
    summary.index = [_label(x) for x in summary.index]
    summary.round(4).to_csv(out_dir / "summary_ensemble_oos_sharpe.csv")

    div_summary = div.groupby("prerun")[["n_factors", "mean_abs_corr", "eff_ratio", "redundancy", "n_clusters"]].mean()
    div_summary.index = [_label(x) for x in div_summary.index]
    div_summary.round(4).to_csv(out_dir / "summary_factor_library_diversity.csv")


def write_index(out_dir: Path, figure_names: list[str]) -> None:
    captions = {
        "fig01_ensemble_oos_sharpe_distribution": (
            "Distribution of ensemble OOS Sharpe values across rolling ETF-month windows. "
            "Boxes omit outlier dots to keep the central distribution visible; black diamonds show means."
        ),
        "fig02_model_robustness_median_oos_sharpe": (
            "Median OOS Sharpe by factor set and model. This highlights whether a factor set performs "
            "only for one model class or remains useful across model specifications."
        ),
        "fig03_ensemble_oos_sharpe_over_time": (
            "Monthly median ensemble OOS Sharpe across ETFs, with the shaded area showing the "
            "interquartile range across ETFs."
        ),
        "fig04_ticker_factor_set_heatmap": (
            "Median ensemble OOS Sharpe by ETF and factor set, summarising which universes drive "
            "the aggregate comparison."
        ),
        "fig05_oos_ic_vs_oos_sharpe": (
            "Relationship between OOS information coefficient and realised ensemble Sharpe. "
            "The Sharpe axis is percentile-clipped for readability."
        ),
        "fig06_factor_library_diversity": (
            "Factor-library diversity summary: effective factor ratio, redundancy, and mean absolute "
            "factor correlation across rolling windows."
        ),
        "fig07_top_factor_importance_recurrence": (
            "Most recurrent top-five factors in the model-based importance analysis. Recurrence is "
            "counted across ticker-month windows and importance models."
        ),
        "fig08_independence_vs_oos_sharpe": (
            "Scatter of factor-library independence versus realised ensemble OOS Sharpe for each "
            "ticker-month and factor set."
        ),
    }

    lines = ["# Rolling factor comparison report figures", ""]
    lines.append("Generated files are written as both `.png` and `.pdf`.")
    lines.append("")
    for name in figure_names:
        lines.append(f"- `{name}.pdf` / `{name}.png`: {captions[name]}")
    lines.append("")
    lines.append("The OOS Sharpe figures are research diagnostics from the frictionless 10-second comparison backtest.")
    (out_dir / "figure_index.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "batch_dir",
        nargs="?",
        default="data/comparisons/rolling_20260624_001308",
        help="Rolling comparison batch directory.",
    )
    args = parser.parse_args()

    batch = Path(args.batch_dir)
    combined = batch / "combined"
    out_dir = batch / "report_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    bf = pd.read_csv(combined / "bruteforce_all.csv")
    div = pd.read_csv(combined / "diversity_all.csv")
    imp = pd.read_csv(combined / "importance_all.csv")

    _setup_style()
    figures = [
        fig_ensemble_distribution(bf, out_dir),
        fig_model_heatmap(bf, out_dir),
        fig_time_trend(bf, out_dir),
        fig_ticker_heatmap(bf, out_dir),
        fig_ic_sharpe_scatter(bf, out_dir),
        fig_diversity(div, out_dir),
        fig_importance_recurrence(imp, out_dir),
        fig_quality_performance(div, bf, out_dir),
    ]
    write_summary_tables(bf, div, out_dir)
    write_index(out_dir, figures)
    print(f"Wrote {len(figures)} figures to {out_dir}")


if __name__ == "__main__":
    main()
