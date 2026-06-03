"""Generate ``fund_showcase.ipynb`` — the supervisor-facing walkthrough.

Run once to (re)build the notebook:  ./venv/bin/python build_showcase_notebook.py
Then execute it (cached mode) with nbconvert, or open it in Jupyter.

The notebook is *cached by default* (RUN_LIVE = False): it loads the artifacts a
``run_fund.py`` run leaves behind (strategy/portfolio DBs, model joblibs, and
``data/strategies/showcase.json``) and renders KPIs, figures and architecture/
workflow diagrams.  Set RUN_LIVE = True in the first cell to re-run the agents.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


# ───────────────────────────────────────────────────────────────────────────
md("""\
# QuantFundAgent — End-to-End Showcase

A LangGraph **multi-agent quant fund** that researches factors, builds ML
strategies, validates them statistically, and allocates capital — mirroring a
systematic trading desk:

**Factor Researcher → Selector → Architect (ML) → Statistician → Portfolio Manager**

This notebook walks the full workflow on 10-second intraday US-equity data with
KPIs, figures and architecture diagrams.  It runs in **cached mode** by default
(loads the artifacts from a `run_fund.py` run); set `RUN_LIVE = True` in the next
cell to drive the agents live.""")

code("""\
from __future__ import annotations
import os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display
warnings.filterwarnings("ignore")

from dotenv import load_dotenv; load_dotenv()
plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": 0.3})

# Cached by default.  True -> re-run the agent pipeline live (needs OPENAI_API_KEY).
RUN_LIVE = False

DATA = Path("data")
from quant_fund_agent.factors import discover_factors
discover_factors()

show = json.load(open(DATA / "strategies" / "showcase.json"))
HORIZON = show["target_horizon"]
print(f"RUN_LIVE={RUN_LIVE} | forecast horizon={HORIZON} bars "
      f"({HORIZON*10}s) | universe={show['n_tickers']} tickers")""")

# ── 1. Architecture ────────────────────────────────────────────────────────
md("## 1. Fund architecture\n\nFive specialised agents share four databases; the "
   "Architect offloads model fitting/backtesting to a **Modeling MCP server** that "
   "owns the heavy data panel.")

code("""\
def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        fc=fc, ec="black", lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=8.5)

def _arrow(ax, x1, y1, x2, y2, color="black"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color))

fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 6)
agents = [(0.1,"Factor\\nResearcher"),(2.1,"Selector\\n(IC-blind)"),
          (4.1,"Architect\\n(ML models)"),(6.3,"Statistician\\n(DSR + OOS)"),
          (8.3,"Portfolio\\nManager")]
for x, label in agents:
    _box(ax, x, 4.5, 1.6, 0.9, label, "#cfe2f3")
for x1, x2 in [(1.7,2.1),(3.7,4.1),(5.9,6.3),(7.9,8.3)]:
    _arrow(ax, x1, 4.95, x2, 4.95)
_box(ax, 3.9, 2.8, 2.2, 0.85, "Modeling MCP server\\n(panel · fit · backtest)", "#fff2cc")
_arrow(ax, 5.0, 4.5, 5.0, 3.65, "tab:orange"); _arrow(ax, 5.1, 3.65, 5.1, 4.5, "tab:orange")
dbs = [(0.1,"Paper DB"),(2.1,"Factor DB"),(4.4,"Strategy DB"),(8.3,"Portfolio DB")]
for x, label in dbs:
    _box(ax, x, 0.4, 1.6, 0.8, label, "#d9ead3")
_arrow(ax, 0.9, 1.2, 0.9, 4.5, "gray"); _arrow(ax, 2.9, 1.2, 2.9, 4.5, "gray")
_arrow(ax, 5.2, 4.5, 5.2, 1.2, "gray"); _arrow(ax, 9.1, 4.5, 9.1, 1.2, "gray")
ax.set_title("Agents · MCP modeling service · shared databases")
plt.show()""")

# ── 2. Workflow ─────────────────────────────────────────────────────────────
md("## 2. Strategy-research workflow\n\nThe Architect runs a **refinement loop**: "
   "design a model → fit on IS-train & score on held-out IS-valid → refit on the full "
   "in-sample window → evaluate → approve or revise.  The Statistician then gates the "
   "candidate on a held-out out-of-sample slice it never saw.")

code("""\
fig, ax = plt.subplots(figsize=(12, 3.0)); ax.axis("off")
ax.set_xlim(0, 12.4); ax.set_ylim(0, 3)
steps = ["Selector\\nhypothesis\\n+ factors", "Architect\\ndesign model",
         "fit + backtest\\n(IS train->valid,\\nrefit full IS)", "evaluate\\napprove / revise",
         "Statistician\\nDSR + OOS gate", "persist +\\nPortfolio Mgr"]
xs = [0.1, 2.2, 4.3, 6.6, 8.7, 10.7]
for x, s in zip(xs, steps):
    _box(ax, x, 1.2, 1.8, 1.0, s, "#cfe2f3")
for i in range(len(xs) - 1):
    _arrow(ax, xs[i] + 1.8, 1.7, xs[i + 1], 1.7)
# revise loop: evaluate -> back to fit
_arrow(ax, xs[3] + 0.4, 1.2, xs[2] + 0.9, 1.2, "tab:red")
ax.text((xs[2] + xs[3]) / 2 + 0.9, 0.8, "revise loop", ha="center", color="tab:red", fontsize=8)
ax.set_title("Selector -> Architect (loop) -> Statistician -> Portfolio Manager")
plt.show()""")

# ── 3. Factor universe ──────────────────────────────────────────────────────
md("## 3. Factor universe & multi-horizon IC (analyst view)\n\nThe alpha factors' "
   "predictive power (rank-IC) **peaks at the 10-second horizon** and decays as the "
   "horizon lengthens.  Note: the Selector is **deliberately blind** to these IC "
   "numbers — it chooses factors on economic role and diversity so the ML model can "
   "exploit interactions (a weak-IC volatility factor can be valuable *combined* with "
   "momentum). This table is for us, the analysts.")

code("""\
fdb = json.load(open(DATA / "factors" / "factor_db.json"))
rows = []
for f in fdb["factors"]:
    by = (f.get("backtest_metrics") or {}).get("ic_by_horizon", {})
    rows.append({"id": f["id"], "name": f["name"], "category": f.get("category", ""),
                 "ic_10s": (by.get("1") or {}).get("ic"),
                 "ic_1m": (by.get("6") or {}).get("ic"),
                 "ic_10m": (by.get("60") or {}).get("ic")})
fac = pd.DataFrame(rows)
print(f"{len(fac)} factors across {fac.category.nunique()} categories")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fac.category.value_counts().plot(kind="bar", ax=ax[0], title="Factors by category", color="#6fa8dc")
ax[0].set_ylabel("count")
fac[["ic_10s", "ic_1m", "ic_10m"]].abs().mean().plot(
    kind="bar", ax=ax[1], title="Mean |IC| by horizon (peaks at 10s)", color="#93c47d")
ax[1].set_ylabel("mean |IC|")
plt.tight_layout(); plt.show()

display(fac.reindex(fac.ic_10s.abs().sort_values(ascending=False).index)
        .head(10).reset_index(drop=True))""")

# ── 4. Selector ─────────────────────────────────────────────────────────────
md("## 4. Selector — IC-blind hypothesis & factor choice\n\nThe Selector forms a "
   "trading hypothesis and picks a *diversified, complementary* set of factors "
   "without seeing any IC scores.")

code("""\
entries = show["entries"]
accepted = [e for e in entries if e.get("approved")]
focus = (accepted[0] if accepted else (entries[0] if entries else None))
assert focus is not None, "No strategy attempts found in showcase.json"
print("Focus strategy is", "ACCEPTED" if focus.get("approved") else f"REJECTED ({focus.get('reject_stage')})")

print("\\nHYPOTHESIS\\n" + "-"*60 + "\\n" + (focus["hypothesis"] or "(n/a)"))
print("\\nSELECTION RATIONALE\\n" + "-"*60 + "\\n" + (focus["selection_rationale"] or "(n/a)"))
chosen = fac[fac.id.isin(focus["factor_ids"])][["id", "name", "category"]].reset_index(drop=True)
print(f"\\nChosen factors ({len(chosen)}) — spanning {chosen.category.nunique()} categories:")
display(chosen)""")

# ── 5. Architect ────────────────────────────────────────────────────────────
md("## 5. Architect — model selection across trials\n\nEach trial fits a model from "
   "the toolbox (regression / random forest / gradient boosting / XGBoost / LightGBM) "
   "and is scored by **held-out validation rank-IC**; the `train_r2` vs `valid_r2` gap "
   "flags overfitting.")

code("""\
trials = pd.DataFrame(focus["trials"])
if not trials.empty:
    cols = [c for c in ["iteration","model_type","n_features","valid_ic","train_r2","valid_r2","is_sharpe"] if c in trials]
    display(trials[cols].round(5))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(trials["iteration"].astype(str) + " · " + trials["model_type"], trials["valid_ic"], color="#6fa8dc")
    ax.set_title("Validation rank-IC by trial (selection criterion)"); ax.set_ylabel("valid IC"); plt.xticks(rotation=20)
    plt.tight_layout(); plt.show()
print("Chosen model:", focus["model_type"], "| params:", focus["model_params"])""")

# ── 6. Performance ──────────────────────────────────────────────────────────
md("## 6. Performance — in-sample vs out-of-sample\n\nThe persisted model is reloaded "
   "(never refit) and back-tested on both the in-sample window and the held-out OOS "
   "slice the Architect never saw.")

code("""\
from quant_fund_agent.modeling import service
from quant_fund_agent.backtesting.data_split import split_panel, split_signals
from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
from quant_fund_agent.strategies.model_strategy import ModelStrategy
from quant_fund_agent.databases import StrategyDatabase

OOS_RATIO = 0.2
ntk = show["n_tickers"]
if isinstance(ntk, int):
    os.environ["ARCHITECT_N_TICKERS"] = str(ntk)

sdb = StrategyDatabase()
try:
    sdb.load_from_json(DATA / "strategies" / "strategy_db.json")
except Exception:
    pass
rec = next((s for s in sdb.list_strategies() if s.id == focus.get("id")), None)
p = (rec.model_params if rec else {}) or {}
artifact = p.get("model_artifact_path") or focus.get("model_artifact_path")
hp = p.get("holding_period") or focus.get("holding_period") or HORIZON
mp = p.get("max_positions") or focus.get("max_positions") or 20
mc = p.get("min_conviction") or focus.get("min_conviction") or 0.0

data_full = service._load_panel()
sig_full = {fid: service._factor_signal(fid, data_full) for fid in focus["factor_ids"]}
data_is, data_oos = split_panel(data_full, OOS_RATIO)
sig_is, sig_oos = split_signals(sig_full, OOS_RATIO)

strat = ModelStrategy.from_artifact(artifact, strategy_id="show", holding_period=hp,
                                    max_positions=mp, min_conviction=mc)
is_bt = backtest_strategy(strat, sig_is, data_is)
oos_bt = backtest_strategy(strat, sig_oos, data_oos)

fig, ax = plt.subplots(figsize=(10, 4))
isc, oosc = is_bt.cumulative_returns, oos_bt.cumulative_returns + is_bt.cumulative_returns.iloc[-1]
ax.plot(isc.index, isc.values, label="In-sample", color="tab:blue")
ax.plot(oosc.index, oosc.values, label="Out-of-sample", color="tab:orange")
ax.axvline(isc.index[-1], ls="--", color="gray")
ax.set_title(f"Cumulative return — {focus.get('name') or focus['model_type']}")
ax.set_ylabel("cumulative return"); ax.legend(); plt.show()

def kpis(m):
    return {"Sharpe": m.sharpe_ratio, "Sortino": m.sortino_ratio, "MaxDD": m.max_drawdown,
            "HitRate": m.hit_rate, "AnnReturn": m.annualised_return,
            "Turnover/day": m.avg_daily_turnover, "IC": m.ic_mean}
display(pd.DataFrame({"In-sample": kpis(is_bt.metrics), "Out-of-sample": kpis(oos_bt.metrics)}).round(4))""")

# ── 7. Statistician ─────────────────────────────────────────────────────────
md("## 7. Statistician — acceptance gates\n\nEvery candidate must clear a **Deflated "
   "Sharpe Ratio** test (corrects for multiple testing & non-normality) and an "
   "**out-of-sample** test (positive OOS Sharpe, bounded decay vs in-sample). "
   "Thresholds are shown explicitly for transparency.")

code("""\
print("Gates:", show.get("gates"))
print("Final decision:", focus.get("final_decision"))
st = pd.DataFrame(focus["stat_tests"])
if not st.empty:
    display(st[["test_id", "verdict", "value", "threshold", "summary"]])

overview = pd.DataFrame([{
    "approved": e.get("approved"), "reject_stage": e.get("reject_stage"),
    "model": e.get("model_type"), "is_sharpe": e.get("is_sharpe"),
    "oos_sharpe": e.get("oos_sharpe"), "deflated_sharpe": e.get("deflated_sharpe"),
} for e in entries])
print("\\nAll attempts this run:")
display(overview.round(4))""")

# ── 8. Portfolio Manager ────────────────────────────────────────────────────
md("## 8. Portfolio Manager — allocation & diversification\n\nThe PM (committee of "
   "risk personalities) allocates capital across the approved book and monitors "
   "cross-strategy correlation.")

code("""\
try:
    pj = json.load(open(DATA / "portfolio" / "portfolio_db.json"))
    recs = pj.get("portfolios") or pj.get("records") or (pj if isinstance(pj, list) else [])
    if isinstance(recs, dict):
        recs = list(recs.values())
    if recs:
        last = recs[-1]
        print("PM:", last.get("pm_name"), "| method:", last.get("construction_method"))
        adf = pd.DataFrame(last.get("allocations", []))
        if not adf.empty:
            keep = [c for c in ["strategy_id", "weight", "enabled", "rationale"] if c in adf]
            display(adf[keep])
    else:
        print("No portfolio records found.")
except Exception as e:
    print("Could not render PM allocation:", e)

try:
    sdb.refresh_correlations()
    corr = sdb.correlation_matrix
    if corr is not None and len(corr) >= 1:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, fontsize=7)
        fig.colorbar(im); ax.set_title("Strategy return correlation"); plt.show()
except Exception as e:
    print("Correlation heatmap unavailable:", e)""")

# ── 9. Horizon comparison ───────────────────────────────────────────────────
md("## 9. Forecast-horizon comparison (10s · 1min · 10min)\n\nWhy trade at 1-minute "
   "when IC peaks at 10s?  A fixed model/feature set is evaluated across horizons: the "
   "**10s horizon has the highest IC but punishing turnover**, while longer horizons "
   "convert more of that signal into realised Sharpe.  (Deterministic; uses ridge.)")

code("""\
from quant_fund_agent.modeling.service import sweep_horizons
sweep = pd.DataFrame(sweep_horizons(factor_ids=focus["factor_ids"], model_type="ridge",
                                    horizons=(1, 6, 60)))
sweep["horizon"] = sweep["target_horizon"].map({1: "10s", 6: "1min", 60: "10min"})
display(sweep[["horizon", "valid_ic", "is_ic", "oos_ic", "is_sharpe", "oos_sharpe"]].round(5))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sweep.set_index("horizon")[["valid_ic", "oos_ic"]].plot(kind="bar", ax=ax[0],
    title="IC by horizon (10s peaks)"); ax[0].set_ylabel("rank-IC")
sweep.set_index("horizon")[["is_sharpe", "oos_sharpe"]].plot(kind="bar", ax=ax[1],
    title="Sharpe by horizon"); ax[1].set_ylabel("Sharpe")
plt.tight_layout(); plt.show()""")

# ── 10. Live ────────────────────────────────────────────────────────────────
md("## 10. (Optional) Run the agents live\n\nSet `RUN_LIVE = True` in the first cell "
   "to drive the real pipeline (Selector → Architect → Statistician) instead of the "
   "cached artifacts.  Requires `OPENAI_API_KEY`.")

code("""\
if RUN_LIVE:
    from quant_fund_agent import pipeline
    res = pipeline.run_strategy_pipeline(max_iterations=2, target_horizon=HORIZON)
    print("approved:", res.approved, "| reject_stage:", res.reject_stage)
    if res.strategy_spec:
        print("model:", res.strategy_spec.model_type, "| factors:", res.strategy_spec.factor_ids)
else:
    print("RUN_LIVE is False — rendered cached results above. Set RUN_LIVE=True to re-run the agents.")""")

# ── 11. Walk-forward backtest ────────────────────────────────────────────────
md("""\
## 11. Walk-forward backtest — running the fund *through time*

Everything above is a single research → strategy → PM pass.  The
**walk-forward backtest** (`quant_fund_agent.simulation`, driven from
`run_backtest.py`) steps the fund over a date range on a **weekly grid**:

- **warm-up** — no meetings or trading until ~1 month of history exists;
- each week the fund holds **research → strategy → PM meetings** (the *same*
  `pipeline.py` functions used in production, so the backtest never forks the
  agent code), then **trades the next week** on data the agents never saw;
- strategies are **frozen on approval** — the PM monitors live PnL and may
  retire them; only new research adds strategies.

**No look-ahead:** at each weekly *cutoff* the Architect/Statistician only see
data strictly *before* it (threaded as `cutoff_date`); the live window is
strictly *after* it.

**Execution & costs (the trade-execution semantics):** per-strategy positions are
combined into the traded book one of two ways (config switch, default *netted*):

- **consolidated netted book** — all live strategies' positions, scaled by the
  PM's capital weights, sum into **one fund book per ticker**, so opposite signals
  on the same name *net out* and agreeing signals reinforce (charged once).
  Fund-level `max_positions`, per-name cap and gross-leverage cap apply to the
  consolidated book.
- **independent pods** — each strategy trades its own book; no netting.

Costs are **spread-aware** (½ the quoted `effSpread`) **+ a commission**, charged
on the traded turnover (no market-impact term).""")

code("""\
# Schematic of the walk-forward loop (reuses _box/_arrow from section 1).
fig, ax = plt.subplots(figsize=(12, 2.8)); ax.axis("off")
ax.set_xlim(0, 12); ax.set_ylim(0, 3)
_box(ax, 0.1, 1.1, 2.0, 1.0, "warm-up\\n(~1 month,\\nno trading)", "#f4cccc")
xs = [2.5, 4.8, 7.1, 9.4]
labels = ["week 1", "week 2", "week 3", "..."]
for x, l in zip(xs, labels):
    _box(ax, x, 1.1, 2.0, 1.0, f"meeting + trade\\n{l}", "#cfe2f3")
_arrow(ax, 2.1, 1.6, 2.5, 1.6)
for i in range(len(xs) - 1):
    _arrow(ax, xs[i] + 2.0, 1.6, xs[i + 1], 1.6)
ax.text(6.0, 0.5, "each week:  research -> strategy -> PM,  then trade [cutoff, next) "
        "as ONE netted book, net of spread+commission costs",
        ha="center", fontsize=8.5, color="#444")
ax.set_title("Walk-forward loop — weekly meetings, frozen strategies, netted execution")
plt.show()""")

md("### Backtest results (latest cached run)\n\nRenders the most recent run under "
   "`data/backtests/<run_id>/`.  If none exists yet, run one first — e.g. "
   "`python run_backtest.py --start 2019-01-02 --end 2019-03-02 --n-tickers 8`.")

code("""\
BT_ROOT = DATA / "backtests"

def _latest_backtest():
    if not BT_ROOT.exists():
        return None
    runs = [d for d in BT_ROOT.iterdir() if d.is_dir() and (d / "fund_metrics.json").exists()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None

run = _latest_backtest()
if run is None:
    print("No backtest run found under data/backtests/.")
    print("Run one first, e.g.:")
    print("  python run_backtest.py --start 2019-01-02 --end 2019-03-02 --n-tickers 8")
    print("  (compare execution models with --execution pod)")
else:
    m = json.load(open(run / "fund_metrics.json"))
    print(f"Backtest run: {run.name}  |  execution: {m.get('execution_model')}")
    kpi_keys = ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
                "annualised_return", "total_return", "total_cost",
                "avg_daily_turnover", "n_strategies_deployed", "n_meetings"]
    display(pd.DataFrame({"value": {k: m.get(k) for k in kpi_keys}}))

    eq = pd.read_csv(run / "equity.csv", parse_dates=["timestamp"]).set_index("timestamp")
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(eq.index, eq["cumulative_return"], color="tab:blue")
    ax[0].set_title(f"Fund cumulative return — {run.name}"); ax[0].set_ylabel("cum. return")
    ax[1].fill_between(eq.index, eq["drawdown"], 0, color="tab:red", alpha=0.4)
    ax[1].set_title("Drawdown"); ax[1].set_ylabel("drawdown")
    plt.tight_layout(); plt.show()

    apath = run / "attribution.csv"
    if apath.exists():
        attr = pd.read_csv(apath, parse_dates=["timestamp"]).set_index("timestamp")
        if not attr.empty:
            fig, ax = plt.subplots(figsize=(11, 4))
            for c in attr.columns:
                ax.plot(attr.index, attr[c], label=c)
            ax.set_title("Per-strategy cumulative contribution (weighted, net of cost)")
            ax.set_ylabel("cum. contribution"); ax.legend(fontsize=7, ncol=2); plt.show()

    mt = [json.loads(l) for l in open(run / "meetings.jsonl") if l.strip()]
    if mt:
        mdf = pd.DataFrame(mt)
        keep = [c for c in ["cutoff", "week", "research", "strategy", "pm",
                            "strategies_approved", "live_bars", "window_return"] if c in mdf]
        print("\\nMeeting log (last 10):")
        display(mdf[keep].tail(10))""")

md("### (Optional) Run a short backtest live\n\nWith `RUN_LIVE = True` this drives a "
   "small walk-forward backtest end-to-end (needs `OPENAI_API_KEY` and `ticker_data/`). "
   "Adjust the dates to your data span.")

code("""\
if RUN_LIVE:
    from quant_fund_agent.simulation import BacktestConfig, run_backtest
    cfg = BacktestConfig(
        start="2019-01-02", end="2019-02-15", warmup="2W",
        initial_strategies=2, n_strategies_per_meeting=1,
        execution_model="netted", n_tickers=8, run_id="showcase_demo",
    )
    bt = run_backtest(cfg)
    print(bt.summary())
else:
    print("RUN_LIVE is False — rendered the latest cached backtest above.")""")


# ───────────────────────────────────────────────────────────────────────────
def build() -> dict:
    nb_cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path("fund_showcase.ipynb")
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")
