"""Persist comparison results: tables (CSV/JSON), figures, a Markdown report, a notebook.

``results`` is a plain dict assembled by ``run_model_comparison.py``::

    {
      "ic_rows": [...], "ic_summary": [...],
      "bruteforce_rows": [...],
      "downstream_summary": [...], "downstream_strategies": [...],
      "usability": {prerun: {"n_usable", "n_dropped", "dropped": {...}}},
      "prerun_models": {prerun: "<research llm>"},
    }

Everything lands under ``cfg.output_dir`` so the whole comparison is one portable
folder to drop into the thesis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from quant_fund_agent.comparison import plots

log = logging.getLogger("comparison.report")


# ── tables ───────────────────────────────────────────────────────────────────

def write_tables(cfg, results: dict[str, Any]) -> dict[str, Path]:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    paths["config"] = out / "config.json"
    (out / "usability.json").write_text(
        json.dumps(results.get("usability", {}), indent=2, default=str))
    paths["usability"] = out / "usability.json"

    for key, fname in [
        ("ic_rows", "ic_results.csv"),
        ("ic_summary", "ic_summary.csv"),
        ("diversity_summary", "analytics_diversity_summary.csv"),
        ("diversity_factor", "analytics_diversity_factor.csv"),
        ("modelview_summary", "analytics_modelview_summary.csv"),
        ("importance_rows", "analytics_importance.csv"),
        ("bruteforce_rows", "bruteforce_results.csv"),
        ("downstream_summary", "downstream_results.csv"),
        ("downstream_strategies", "downstream_strategies.csv"),
    ]:
        rows = results.get(key) or []
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out / fname, index=False)
            paths[key] = out / fname
            log.info("wrote table %s (%d rows)", out / fname, len(df))

    # Per-prerun correlation matrices (DataFrames, written with index).
    for prerun, corr in (results.get("corr_matrices") or {}).items():
        if corr is not None and not getattr(corr, "empty", True):
            fp = out / f"factor_correlation_{prerun}.csv"
            corr.to_csv(fp)
            paths[f"corr_{prerun}"] = fp
    return paths


# ── figures ──────────────────────────────────────────────────────────────────

def render_figures(cfg, results: dict[str, Any]) -> dict[str, Path]:
    fd = cfg.figures_dir
    figs: dict[str, Path] = {}

    def _add(key, p):
        if p is not None:
            figs[key] = p

    if results.get("usability"):
        _add("factor_usability", plots.factor_usability(results["usability"], fd))
    if results.get("ic_rows"):
        _add("ic_distribution", plots.ic_distribution(results["ic_rows"], fd, h=cfg.target_horizon))
        _add("ic_top_factors", plots.ic_top_factors(results["ic_rows"], fd, h=cfg.target_horizon))
    if results.get("ic_summary"):
        _add("ic_mean_by_horizon",
             plots.ic_mean_by_horizon(results["ic_summary"], fd, tuple(cfg.ic_horizons)))
    if results.get("diversity_summary"):
        _add("effective_factors", plots.effective_factors_bar(results["diversity_summary"], fd))
    for prerun, corr in (results.get("corr_matrices") or {}).items():
        _add(f"corr_{prerun}", plots.factor_correlation_heatmap(corr, fd, prerun))
    if results.get("modelview_summary"):
        _add("deflation", plots.deflation_bar(results["modelview_summary"], fd))
    if results.get("importance_rows"):
        for model in dict.fromkeys(r["model"] for r in results["importance_rows"]):
            _add(f"feature_importance_{model}",
                 plots.feature_importance_bar(results["importance_rows"], fd, model=model))
    if results.get("bruteforce_rows"):
        _add("bruteforce_oos_sharpe", plots.bruteforce_oos_sharpe(results["bruteforce_rows"], fd))
        _add("bruteforce_ic_heatmap", plots.bruteforce_ic_heatmap(results["bruteforce_rows"], fd))
        _add("bruteforce_is_vs_oos", plots.bruteforce_is_vs_oos(results["bruteforce_rows"], fd))
    if results.get("downstream_summary"):
        _add("downstream_summary", plots.downstream_summary(results["downstream_summary"], fd))
    return figs


# ── markdown report ──────────────────────────────────────────────────────────

def _md_table(rows: list[dict[str, Any]], cols: list[str], round_to: int = 4) -> str:
    if not rows:
        return "_(no data)_\n"
    df = pd.DataFrame(rows)
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    for c in cols:
        df[c] = df[c].map(lambda v: round(v, round_to) if isinstance(v, float) else v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |"
                     for r in df.itertuples(index=False, name=None))
    return f"{head}\n{sep}\n{body}\n"


def _findings(results: dict[str, Any]) -> list[str]:
    out: list[str] = []
    bf = pd.DataFrame(results.get("bruteforce_rows") or [])
    if not bf.empty and "oos_sharpe" in bf and bf["oos_sharpe"].notna().any():
        best = bf.loc[bf["oos_sharpe"].idxmax()]
        out.append(f"**Best ML-combined OOS Sharpe:** `{best['prerun']}` with "
                   f"`{best['model']}` (OOS Sharpe = {best['oos_sharpe']:.3f}).")
        per = bf.dropna(subset=["oos_sharpe"]).groupby("prerun")["oos_sharpe"].mean()
        if len(per):
            out.append("**Mean OOS Sharpe across models, by research set:** "
                       + ", ".join(f"`{p}` = {v:.3f}" for p, v in per.sort_values(ascending=False).items()) + ".")
    ic = pd.DataFrame(results.get("ic_summary") or [])
    if not ic.empty and "mean_abs_ic_6" in ic and ic["mean_abs_ic_6"].notna().any():
        best = ic.loc[ic["mean_abs_ic_6"].idxmax()]
        out.append(f"**Highest mean single-factor |IC| (h=6):** `{best['prerun']}` "
                   f"(mean |IC| = {best['mean_abs_ic_6']:.4f}).")
    dv = pd.DataFrame(results.get("diversity_summary") or [])
    if not dv.empty and "eff_ratio" in dv and dv["eff_ratio"].notna().any():
        best = dv.loc[dv["eff_ratio"].idxmax()]
        out.append(f"**Most diverse zoo (highest effective/raw factor ratio):** "
                   f"`{best['prerun']}` (eff {best['eff_n_factors']:.1f} of "
                   f"{int(best['n_factors'])}, ratio {best['eff_ratio']:.2f}).")
    mv = pd.DataFrame(results.get("modelview_summary") or [])
    if not mv.empty and "deflated_best_ic" in mv and mv["deflated_best_ic"].notna().any():
        best = mv.loc[mv["deflated_best_ic"].idxmax()]
        out.append(f"**Best selection-deflated single-factor |IC|:** `{best['prerun']}` "
                   f"(deflated |IC| = {best['deflated_best_ic']:.4f} from "
                   f"{int(best['ic_n_tested'])} factors tried).")
    ds = pd.DataFrame(results.get("downstream_summary") or [])
    if not ds.empty and "mean_oos_sharpe" in ds and ds["mean_oos_sharpe"].notna().any():
        best = ds.loc[ds["mean_oos_sharpe"].idxmax()]
        out.append(f"**Best downstream portfolio (approved-strategy mean OOS Sharpe):** "
                   f"`{best['prerun']}` ({best['mean_oos_sharpe']:.3f}).")
    return out


def write_report_md(cfg, results: dict[str, Any], figures: dict[str, Path]) -> Path:
    out = cfg.output_dir
    pm = results.get("prerun_models", {})
    lines: list[str] = []

    def fig(key: str, caption: str) -> None:
        p = figures.get(key)
        if p is not None:
            lines.append(f"\n![{caption}](figures/{p.name})\n\n*{caption}*\n")

    lines.append(f"# Research-LLM factor comparison — `{cfg.comparison_id}`\n")
    lines.append("Comparing the **factor output of different research LLMs** on three axes "
                 "(raw factor quality → factors as ML features → factors through the full "
                 "agentic fund). The panel, IS/OOS split, horizon and model hyper-parameters "
                 "are held identical across preruns — the *only* variable is the factor set.\n")

    lines.append("## Preruns compared\n")
    pre_rows = [{"prerun": p, "research_model": pm.get(p, "?"),
                 "usable_factors": (results.get("usability", {}).get(p, {}) or {}).get("n_usable"),
                 "dropped_factors": (results.get("usability", {}).get(p, {}) or {}).get("n_dropped")}
                for p in cfg.resolved_preruns()]
    lines.append(_md_table(pre_rows, ["prerun", "research_model", "usable_factors", "dropped_factors"]))
    lines.append("> Dropped factors declare fields the current data doesn't have yet "
                 "(e.g. fundamentals); they light up unchanged once that data is downloaded.\n")
    fig("factor_usability", "Usable vs awaiting-data factors per research model")

    findings = _findings(results)
    if findings:
        lines.append("## Key findings\n")
        lines.extend(f"- {f}" for f in findings)
        lines.append("")

    if results.get("ic_summary"):
        _ic_kind = ("per-underlying time-series" if getattr(cfg, "fit_standardize", "per_underlying")
                    != "cross_sectional" else "cross-sectional")
        lines.append("## 1. Single-factor IC (raw factor quality)\n")
        lines.append(f"{_ic_kind.capitalize()} Pearson IC of every researched factor, "
                     "recomputed on the shared panel at horizons "
                     + ", ".join(f"h={h}" for h in cfg.ic_horizons) + ". The "
                     "per-underlying IC computes one factor/forward-return correlation "
                     "per asset and aggregates them by valid observation count; "
                     "the cross-sectional IC correlates across underlyings per timestamp.\n")
        lines.append(_md_table(results["ic_summary"],
                               ["prerun", "n_factors", "mean_abs_ic_1", "mean_abs_ic_6",
                                "mean_abs_ic_60", "mean_abs_icir_6", "best_factor_h6", "best_abs_ic_h6"]))
        fig("ic_mean_by_horizon", "Mean |IC| by research model and horizon")
        fig("ic_distribution", "Per-factor |IC| distribution by research model")
        fig("ic_top_factors", "Top factors by |IC| per research model")

    if results.get("diversity_summary"):
        lines.append("## 2. Factor diversity & redundancy\n")
        lines.append("Pairwise correlation of each zoo's *signals*. `eff_n_factors` is the "
                     "effective number of independent factors (participation ratio of the "
                     "correlation eigenvalues); `eff_ratio` and `redundancy` summarise how "
                     "much unique information the zoo holds vs. how much is duplicated; "
                     "`n_clusters` groups factors at |corr| ≥ "
                     f"{cfg.corr_threshold}.\n")
        lines.append(_md_table(results["diversity_summary"],
                               ["prerun", "n_factors", "eff_n_factors", "eff_ratio",
                                "mean_abs_corr", "n_clusters", "redundancy"]))
        fig("effective_factors", "Effective vs raw factor count per research model")
        for p in cfg.resolved_preruns():
            fig(f"corr_{p}", f"Signal correlation matrix — {p}")

    if results.get("modelview_summary"):
        lines.append("## 3. Deflation & model-based importance\n")
        lines.append("`deflated_best_ic` haircuts each zoo's best |IC| for the number of "
                     "factors tried (`ic_n_tested`) — a bigger zoo's best factor is more "
                     "likely to be lucky. `lasso_n_nonzero` / `lasso_sparsity` show how many "
                     "factors a sparse linear model actually keeps (model-view redundancy).\n")
        lines.append(_md_table(results["modelview_summary"],
                               ["prerun", "best_ic", "deflated_best_ic", "deflated_best_t",
                                "ic_n_tested", "ic_n_obs", "lasso_n_nonzero", "lasso_sparsity"]))
        fig("deflation", "Best |IC| before vs after multiple-testing deflation")
        for model in dict.fromkeys(r["model"] for r in (results.get("importance_rows") or [])):
            fig(f"feature_importance_{model}", f"Top factors by {model} importance per zoo")

    if results.get("bruteforce_rows"):
        lines.append("## 4. ML-combined signal — per-underlying vectorised backtest\n")
        lines.append("Each model combines a prerun's factors into ONE signal (fit "
                     "`factors → forward return` on IS, predict per (bar, underlying)), then "
                     "that combined signal is run through a simple vectorised backtest — "
                     "`position(signal) × the underlying's own forward return` — on the "
                     "held-out OOS tail (+ an equal-weight ensemble). No cross-sectional "
                     "ranking.\n")
        lines.append(f"> Config: position=**{cfg.position_mode}** (t={cfg.position_threshold}, "
                     f"z-score `{cfg.position_zscore_basis}`), aggregation=**{cfg.backtest_aggregation}**, "
                     f"fit-standardise=**{cfg.fit_standardize}**, horizon={cfg.target_horizon}.\n")
        lines.append(_md_table(results["bruteforce_rows"],
                               ["prerun", "model", "n_factors_used", "oos_ic", "oos_sharpe",
                                "is_sharpe", "oos_ann_return", "oos_max_drawdown"]))
        fig("bruteforce_oos_sharpe", "ML-combined OOS Sharpe by model and research set")
        fig("bruteforce_ic_heatmap", "ML-combined OOS IC (prerun × model)")
        fig("bruteforce_is_vs_oos", "IS vs OOS Sharpe (overfitting diagnostic)")

    if results.get("downstream_summary"):
        lines.append("## 5. Downstream agents (the full fund)\n")
        lines.append("Each prerun's factors run through Selector → Architect → Statistician → PM; "
                     "per-strategy OOS verdicts and the PM's expected portfolio metrics.\n")
        lines.append(_md_table(results["downstream_summary"],
                               ["prerun", "n_factors", "n_approved", "n_attempted",
                                "mean_oos_sharpe", "portfolio_expected_sharpe"]))
        fig("downstream_summary", "Downstream approved-strategy OOS Sharpe and counts")

    lines.append("\n---\n*Generated by `run_model_comparison.py`. Tables: `*.csv`; "
                 "interactive: `comparison.ipynb`.*\n")

    path = out / "report.md"
    path.write_text("\n".join(lines))
    log.info("wrote report %s", path)
    return path


# ── notebook ─────────────────────────────────────────────────────────────────

def build_comparison_notebook(cfg) -> Path:
    """Generate ``comparison.ipynb`` that re-renders the comparison from the CSVs."""
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    cells: list[Any] = []
    cells.append(new_markdown_cell(
        f"# Research-LLM factor comparison — `{cfg.comparison_id}`\n\n"
        "Cached view of the comparison: loads the CSV tables this run wrote and "
        "re-renders the figures inline. Three tracks — single-factor IC, brute-force "
        "ML, and the downstream agentic fund — across preruns mined by different "
        "research LLMs."))
    cells.append(new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from quant_fund_agent.comparison import plots\n"
        "HERE = Path.cwd()\n"
        "cfg = json.load(open(HERE / 'config.json'))\n"
        "FIG = HERE / 'figures'\n"
        "def load(name):\n"
        "    p = HERE / name\n"
        "    return pd.read_csv(p) if p.exists() else pd.DataFrame()\n"
        "print('comparison', cfg['comparison_id'], '| preruns:', cfg['resolved_preruns'])"))

    cells.append(new_markdown_cell("## Factor usability on the current data"))
    cells.append(new_code_cell(
        "usability = json.load(open(HERE / 'usability.json'))\n"
        "pd.DataFrame([{'prerun': k, **{kk: vv for kk, vv in v.items() if kk != 'dropped'}}\n"
        "              for k, v in usability.items()])"))

    cells.append(new_markdown_cell("## 1. Single-factor IC"))
    cells.append(new_code_cell(
        "ic_summary = load('ic_summary.csv'); ic_rows = load('ic_results.csv')\n"
        "display(ic_summary)\n"
        "for f in ['ic_mean_by_horizon.png', 'ic_distribution_h%d.png' % cfg['target_horizon'],\n"
        "          'ic_top_factors_h%d.png' % cfg['target_horizon']]:\n"
        "    p = FIG / f\n"
        "    if p.exists():\n"
        "        display(plt.imread(p).shape); from IPython.display import Image, display as d; d(Image(str(p)))"))

    cells.append(new_markdown_cell("## 2. Factor diversity & redundancy"))
    cells.append(new_code_cell(
        "div = load('analytics_diversity_summary.csv'); display(div)\n"
        "from IPython.display import Image, display as d\n"
        "p = FIG / 'effective_factors.png'\n"
        "if p.exists(): d(Image(str(p)))\n"
        "for pr in cfg['resolved_preruns']:\n"
        "    q = FIG / ('factor_correlation_%s.png' % pr)\n"
        "    if q.exists(): d(Image(str(q)))"))

    cells.append(new_markdown_cell("## 3. Deflation & model-based importance"))
    cells.append(new_code_cell(
        "mv = load('analytics_modelview_summary.csv'); display(mv)\n"
        "imp = load('analytics_importance.csv'); display(imp)\n"
        "from IPython.display import Image, display as d\n"
        "for f in ['deflation.png'] + sorted(str(x.name) for x in FIG.glob('feature_importance_*.png')):\n"
        "    p = FIG / f\n"
        "    if p.exists(): d(Image(str(p)))"))

    cells.append(new_markdown_cell("## 4. ML-combined signal — vectorised backtest"))
    cells.append(new_code_cell(
        "bf = load('bruteforce_results.csv'); display(bf)\n"
        "from IPython.display import Image, display as d\n"
        "for f in ['bruteforce_oos_sharpe.png', 'bruteforce_ic_heatmap.png', 'bruteforce_is_vs_oos.png']:\n"
        "    p = FIG / f\n"
        "    if p.exists(): d(Image(str(p)))"))

    cells.append(new_markdown_cell("## 5. Downstream agents"))
    cells.append(new_code_cell(
        "ds = load('downstream_results.csv'); display(ds)\n"
        "from IPython.display import Image, display as d\n"
        "p = FIG / 'downstream_summary.png'\n"
        "if p.exists(): d(Image(str(p)))"))

    nb = new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    path = cfg.output_dir / "comparison.ipynb"
    nbformat.write(nb, str(path))
    log.info("wrote notebook %s", path)
    return path
