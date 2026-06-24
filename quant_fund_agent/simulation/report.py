"""Presentation-ready analytics for a walk-forward backtest.

Given a finalised :class:`~quant_fund_agent.simulation.results.BacktestResults`,
:func:`render_report` writes a folder of PNG figures and a ``report.md`` that
embeds them next to a numeric KPI table — the visual + numeric summary the fund
needs (NAV curve, drawdown, percent invested, names held, rolling Sharpe, monthly
returns, per-strategy attribution, and — in prerun-inject mode — the catalog
growth over time).

Everything is best-effort and self-contained: matplotlib runs headless (Agg) and
each figure is wrapped so one empty/short series never aborts the rest of the
report.  No LLM, no network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_fund_agent.backtesting.strategy_backtester import _drawdown_series
from quant_fund_agent.data.frequency import (
    bars_per_day_from_index,
    trading_days_per_year_from_index,
)

log = logging.getLogger("simulation.report")


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    return float(bars_per_day_from_index(index) * trading_days_per_year_from_index(index))


def render_report(results, out_dir: Path) -> Path:
    """Write figures + ``report.md`` under ``out_dir/report`` and return its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "report"
    fig_dir.mkdir(parents=True, exist_ok=True)

    r = results.fund_returns
    if r is None or len(r) == 0:
        log.warning("render_report: no fund returns — nothing to plot.")
        (fig_dir / "report.md").write_text("# Backtest report\n\n_No traded bars._\n")
        return fig_dir / "report.md"

    figures: list[tuple[str, str]] = []  # (title, filename)

    def _save(fig, name: str, title: str) -> None:
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figures.append((title, name))

    def _safe(fn, name: str, title: str) -> None:
        try:
            fn(name, title)
        except Exception as e:  # noqa: BLE001 — one bad chart must not kill the report
            log.warning("report: figure %r failed: %s", name, e)

    nav = results.nav.reindex(r.index)
    cum = r.cumsum()
    dd = _drawdown_series(cum)

    # 1 ── NAV + drawdown (two stacked panels) ─────────────────────────────
    def _nav(name, title):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(nav.index, nav.values, color="#1f4e79", lw=1.4)
        ax1.set_ylabel("NAV ($)")
        ax1.set_title(f"Fund NAV — {results.config.run_id}")
        ax1.grid(alpha=0.3)
        ax2.fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.5)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(alpha=0.3)
        _save(fig, name, title)
    _safe(_nav, "nav_drawdown.png", "NAV & drawdown")

    # 2 ── cumulative return ───────────────────────────────────────────────
    def _cum(name, title):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(cum.index, cum.values * 100, color="#1f4e79", lw=1.4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("Cumulative return (%)")
        ax.set_title("Cumulative return (additive)")
        ax.grid(alpha=0.3)
        _save(fig, name, title)
    _safe(_cum, "cumulative_return.png", "Cumulative return")

    # 3 ── percent invested + names held (twin axis) ───────────────────────
    expo = results.gross_exposure.reindex(r.index)
    npos = results.n_positions.reindex(r.index)

    def _invested(name, title):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(expo.index, expo.values * 100, color="#117a65", lw=1.2, label="gross exposure")
        ax.set_ylabel("% invested (gross exposure)", color="#117a65")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(npos.index, npos.values, color="#b9770e", lw=0.9, alpha=0.7, label="names held")
        ax2.set_ylabel("# names held", color="#b9770e")
        ax.set_title("Capital deployed over time")
        _save(fig, name, title)
    _safe(_invested, "percent_invested.png", "Percent invested & names held")

    # 4 ── rolling annualised Sharpe ───────────────────────────────────────
    def _rolling_sharpe(name, title):
        bpy = _bars_per_year(r.index)
        win = max(20, int(round(bpy / 4)))  # ~one quarter
        if len(r) < win + 5:
            raise ValueError("series too short for rolling Sharpe")
        roll = r.rolling(win)
        sharpe = (roll.mean() / roll.std()).replace([np.inf, -np.inf], np.nan) * np.sqrt(bpy)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sharpe.index, sharpe.values, color="#6c3483", lw=1.1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("Annualised Sharpe")
        ax.set_title(f"Rolling Sharpe ({win}-bar window)")
        ax.grid(alpha=0.3)
        _save(fig, name, title)
    _safe(_rolling_sharpe, "rolling_sharpe.png", "Rolling Sharpe")

    # 5 ── monthly returns bar chart ───────────────────────────────────────
    def _monthly(name, title):
        monthly = r.resample("ME").sum()
        if len(monthly) < 2:
            raise ValueError("not enough months")
        colors = ["#117a65" if v >= 0 else "#c0392b" for v in monthly.values]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(range(len(monthly)), monthly.values * 100, color=colors)
        step = max(1, len(monthly) // 12)
        ax.set_xticks(range(0, len(monthly), step))
        ax.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index[::step]],
                           rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("Monthly return (%)")
        ax.set_title("Monthly returns")
        _save(fig, name, title)
    _safe(_monthly, "monthly_returns.png", "Monthly returns")

    # 6 ── per-strategy attribution (cumulative) ───────────────────────────
    def _attribution(name, title):
        attr = results.attribution
        if attr is None or attr.empty:
            raise ValueError("no attribution")
        cumattr = attr.cumsum()
        totals = cumattr.iloc[-1].abs().sort_values(ascending=False)
        top = list(totals.head(12).index)
        fig, ax = plt.subplots(figsize=(11, 5))
        for sid in top:
            ax.plot(cumattr.index, cumattr[sid].values * 100, lw=1.0, label=sid[:24])
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("Cumulative contribution (%)")
        ax.set_title(f"Per-strategy attribution (top {len(top)})")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(alpha=0.3)
        _save(fig, name, title)
    _safe(_attribution, "attribution.png", "Per-strategy attribution")

    # 7 ── catalog growth: factors & strategies over meetings ──────────────
    def _growth(name, title):
        rows = []
        cum_fac, cum_strat = 0, 0
        for mtg in results.meetings:
            cum_fac += len(mtg.get("injected_factor_ids") or mtg.get("kept_factor_ids") or [])
            cum_strat += len(mtg.get("strategies_approved") or [])
            rows.append((pd.Timestamp(mtg["cutoff"]), cum_fac, cum_strat))
        if not rows:
            raise ValueError("no meetings")
        idx = [t for t, _, _ in rows]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.step(idx, [f for _, f, _ in rows], where="post", color="#1f4e79",
                lw=1.3, label="factors in catalog")
        ax.step(idx, [s for _, _, s in rows], where="post", color="#b9770e",
                lw=1.3, label="strategies approved")
        ax.set_ylabel("cumulative count")
        ax.set_title("Catalog & strategy growth over the backtest")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, name, title)
    _safe(_growth, "catalog_growth.png", "Catalog & strategy growth")

    # ── report.md ──────────────────────────────────────────────────────────
    md = _build_markdown(results, figures)
    (fig_dir / "report.md").write_text(md)
    log.info("Backtest report → %s (%d figures)", fig_dir / "report.md", len(figures))
    return fig_dir / "report.md"


def _build_markdown(results, figures: list[tuple[str, str]]) -> str:
    m: dict[str, Any] = results.metrics or {}
    cfg = results.config

    def g(k, default="—"):
        v = m.get(k)
        return default if v is None else v

    lines = [
        f"# Walk-forward backtest — {cfg.run_id}",
        "",
        f"- **Span**: {cfg.start} → {cfg.end}",
        f"- **Execution**: {g('execution_model')} book",
        f"- **Factor source**: {cfg.factor_source}"
        + (f" (preruns: {', '.join(cfg.inject_preruns)}, "
           f"{cfg.factors_per_inject}/meeting)" if cfg.factor_source == "prerun_inject" else ""),
        f"- **Meetings**: {g('n_meetings')}  |  **strategies deployed**: {g('n_strategies_deployed')}",
        "",
        "## Key metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Final NAV | {g('final_nav')} (capital {cfg.capital:,.0f}) |",
        f"| Total return | {g('total_return')} |",
        f"| Annualised return | {g('annualised_return')} |",
        f"| Annualised volatility | {g('annualised_volatility')} |",
        f"| Sharpe | {g('sharpe_ratio')} |",
        f"| Sortino | {g('sortino_ratio')} |",
        f"| Calmar | {g('calmar_ratio')} |",
        f"| Max drawdown | {g('max_drawdown')} |",
        f"| Max DD duration (bars) | {g('max_drawdown_duration_bars')} |",
        f"| Hit rate | {g('hit_rate')} |",
        f"| % invested (mean) | {g('pct_invested_mean')} |",
        f"| % invested (max) | {g('pct_invested_max')} |",
        f"| Avg names held | {g('n_positions_mean')} |",
        f"| Bars traded | {g('n_bars')} |",
        f"| Total cost (return) | {g('total_cost')} |",
        f"| Avg daily turnover | {g('avg_daily_turnover')} |",
        "",
        "## Figures",
        "",
    ]
    for title, fname in figures:
        lines += [f"### {title}", "", f"![{title}]({fname})", ""]
    return "\n".join(lines)
