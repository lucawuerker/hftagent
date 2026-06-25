"""Automate a rolling-window factor-set comparison over the LOBSTER tickers.

For **every ticker** under ``ticker_data/`` this runs ``run_model_comparison.py``
on a **rolling IS/OOS month window** (2 IS months + the next OOS month by default,
stepping one month forward each time), comparing the configured factor-research
preruns *per underlying* (no cross-section), then **aggregates** every run into
combined tables + a per-ticker *feature-importance-over-OOS-months* analysis under
``data/comparisons/<batch>/``.

Memory safety: each (ticker, window) runs in its **own subprocess**, so the OS
reclaims its memory before the next one — the large intraday panel never
accumulates.  The sweep is resumable (completed windows are skipped) and robust
(a failed run is logged; the sweep continues).

Examples
--------
::

    # The whole sweep: all tickers, full resolution, all models (the default).
    ./venv/bin/python run_rolling_comparison.py

    # Quick smoke on one ticker, capped for speed.
    ./venv/bin/python run_rolling_comparison.py --tickers CORN --max-bars 5000 --name smoke

    # Re-build the combined tables / figures without re-running anything.
    ./venv/bin/python run_rolling_comparison.py --name smoke --aggregate-only
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_rolling_comparison")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"),
                   help="LOBSTER per-ticker CSV root (default ticker_data).")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated tickers (default: every dir under --data-dir).")
    p.add_argument("--preruns", default=None,
                   help="Comma-separated prerun names to compare "
                        "(default: gpt4omini120650,gpt5.4mini120650,main).")
    p.add_argument("--name", default=None,
                   help="Batch name → data/comparisons/<name>/ (default rolling_<timestamp>).")
    p.add_argument("--out-root", default="data/comparisons",
                   help="Root under which the batch folder is written.")
    # rolling window shape
    p.add_argument("--is-len", type=int, default=2, help="In-sample months per window (default 2).")
    p.add_argument("--oos-len", type=int, default=1, help="Out-of-sample months per window (default 1).")
    p.add_argument("--step", type=int, default=1, help="Months to advance per window (default 1).")
    # speed / fidelity (passthrough to run_model_comparison.py)
    p.add_argument("--models", default=None,
                   help="Restrict the brute-force model catalog (default: all models).")
    p.add_argument("--max-bars", type=int, default=None,
                   help="Stride each window's panel to ≤ N bars (default: full resolution).")
    p.add_argument("--importance-top-n", type=int, default=200,
                   help="Top factors kept per (prerun, importance model) — high so the FULL "
                        "per-factor importance vector is kept for the over-months matrices "
                        "(default 200).")
    # execution
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel windows (each a subprocess). Default 1 (serial). >1 multiplies "
                        "peak memory by --jobs.")
    p.add_argument("--force", action="store_true",
                   help="Re-run windows even if their status.json shows them complete.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip running; just (re)build combined tables + per-ticker figures.")
    return p.parse_args()


def _resolve_tickers(data_dir: str, spec: str | None) -> list[str]:
    if spec:
        return [t.strip().upper() for t in spec.split(",") if t.strip()]
    root = Path(data_dir)
    if not root.is_dir():
        raise SystemExit(f"--data-dir {data_dir!r} is not a directory.")
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any(d.glob("bin*.csv")))


# ── cross-ticker figures (OOS metrics over time + Sharpe-vs-volume) ──────────
#
# These read the aggregated ``combined/bruteforce_all.csv`` (keyed by
# ``ticker, oos_month, prerun, model``) and the raw ``ticker_data`` CSVs, and are
# rebuilt automatically at the end of every sweep (and on ``--aggregate-only``).

# Coarse asset-class "sector" for each ETF — used only to colour the
# Sharpe-vs-volume scatter.  Edit/extend this map to re-bucket the universe;
# unmapped tickers fall back to "Other".
TICKER_SECTORS = {
    "SPY": "Equity", "QQQ": "Equity", "FXI": "Equity",   # stock-index ETFs
    "IEF": "Rates", "SHY": "Rates",                       # Treasury ETFs
    "GLD": "Commodity", "USO": "Commodity",               # gold / oil
    "CORN": "Commodity", "CPER": "Commodity",             # corn / copper
    "FXE": "FX",                                          # euro
}


def _average_daily_volume(data_dir: str, tickers: list[str]) -> dict[str, float]:
    """Average daily traded volume per ticker (mean over days of summed ``trdLiq``).

    ``trdLiq`` is the per-bar sum of executed trade sizes (shares), so summing it
    over a session's bars gives that day's traded volume.  Only ``date`` +
    ``trdLiq`` are read, so this stays cheap on the large intraday CSVs.
    """
    import pandas as pd

    adv: dict[str, float] = {}
    for t in tickers:
        daily = []
        for csv in sorted((Path(data_dir) / t).glob("bin*.csv")):
            try:
                df = pd.read_csv(csv, usecols=["date", "trdLiq"])
            except Exception:  # noqa: BLE001 — skip a truncated/odd file
                continue
            if not df.empty:
                daily.append(df.groupby("date")["trdLiq"].sum())
        if daily:
            allday = pd.concat(daily)
            pos = allday[allday > 0]
            adv[t] = float(pos.mean()) if not pos.empty else float(allday.mean())
    return adv


def _plot_lines_over_months(g, value_col: str, color_for: dict, *,
                            title: str, ylabel: str, out_path: Path):
    """One coloured line per ETF: ``value_col`` over the OOS months."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    piv = g.pivot_table(index="oos_month", columns="ticker", values=value_col, aggfunc="mean")
    if piv.empty or int(piv.notna().to_numpy().sum()) == 0:
        return None
    piv = piv.reindex(sorted(piv.index))
    x = pd.to_datetime([f"{m}-01" for m in piv.index])
    fig, ax = plt.subplots(figsize=(max(7.0, 0.4 * len(piv.index) + 3), 4.5))
    for t in piv.columns:
        ax.plot(x, piv[t], marker="o", ms=3.5, lw=1.4, color=color_for.get(t), label=str(t))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("OOS month")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2, title="ETF")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_sharpe_vs_volume(bf, data_dir: str, sectors: dict[str, str], out_path: Path):
    """Scatter: mean OOS Sharpe (y) vs average daily volume (x), one dot per ETF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    perf = bf.dropna(subset=["oos_sharpe"]).groupby("ticker")["oos_sharpe"].mean()
    if perf.empty:
        return None
    adv = _average_daily_volume(data_dir, list(perf.index))
    rows = [(t, adv[t], float(perf[t]), sectors.get(t, "Other"))
            for t in perf.index if t in adv and adv[t] > 0]
    if not rows:
        return None
    secs = sorted({r[3] for r in rows})
    cmap = plt.get_cmap("Set1")
    sec_color = {s: cmap(i % 9) for i, s in enumerate(secs)}
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for s in secs:
        pts = [r for r in rows if r[3] == s]
        ax.scatter([r[1] for r in pts], [r[2] for r in pts], s=110, color=sec_color[s],
                   label=s, edgecolor="k", linewidth=0.6, alpha=0.9, zorder=3)
    for t, vol, shp, _ in rows:
        ax.annotate(t, (vol, shp), fontsize=8, xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Average daily volume (shares traded, log scale)")
    ax.set_ylabel("Mean OOS Sharpe (all factor sets & models)")
    ax.set_title("Mean OOS Sharpe vs average daily volume, by ETF")
    ax.legend(title="Sector", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_cross_ticker_figures(batch_dir: Path, data_dir: str,
                               sectors: dict[str, str] | None = None) -> list[Path]:
    """Cross-ticker figures from ``combined/bruteforce_all.csv`` → ``cross_ticker/``.

    Per factor set (``prerun``): OOS Sharpe, combined-signal OOS IC and the OOS÷IS
    Sharpe ratio over the OOS months — **one coloured line per ETF** (each metric
    averaged across the brute-force models).  Plus a single Sharpe-vs-volume
    scatter — mean OOS Sharpe vs average daily volume, one dot per ETF, coloured
    by sector.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    sectors = sectors or TICKER_SECTORS
    bf_path = batch_dir / "combined" / "bruteforce_all.csv"
    if not bf_path.exists():
        log.warning("no %s — skipping cross-ticker figures", bf_path)
        return []
    bf = pd.read_csv(bf_path)
    if bf.empty:
        return []

    out_dir = batch_dir / "cross_ticker"
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(bf["ticker"].dropna().unique())
    cmap = plt.get_cmap("tab10")
    color_for = {t: cmap(i % 10) for i, t in enumerate(tickers)}

    # Per (prerun, oos_month, ticker): mean across models of each metric.
    grp = (bf.groupby(["prerun", "oos_month", "ticker"])
             .agg(oos_sharpe=("oos_sharpe", "mean"),
                  is_sharpe=("is_sharpe", "mean"),
                  oos_ic=("oos_ic", "mean"))
             .reset_index())
    grp["oos_is_sharpe_ratio"] = grp["oos_sharpe"] / grp["is_sharpe"].replace(0, float("nan"))

    metrics = [
        ("oos_sharpe", "OOS Sharpe", "OOS Sharpe (mean across models)"),
        ("oos_ic", "OOS IC", "combined-signal OOS IC (mean across models)"),
        ("oos_is_sharpe_ratio", "OOS / IS Sharpe", "OOS ÷ IS Sharpe (mean across models)"),
    ]
    paths: list[Path] = []
    for prerun, g in grp.groupby("prerun"):
        for col, short, ylab in metrics:
            p = _plot_lines_over_months(
                g, col, color_for,
                title=f"{short} over OOS months by ETF — factor set '{prerun}'",
                ylabel=ylab, out_path=out_dir / f"{col}_over_time__{prerun}.png")
            if p is not None:
                paths.append(p)

    p = _plot_sharpe_vs_volume(bf, data_dir, sectors,
                               out_dir / "sharpe_vs_volume_by_sector.png")
    if p is not None:
        paths.append(p)

    log.info("cross-ticker figures → %s (%d files)", out_dir, len(paths))
    return paths


def main() -> None:
    args = _parse_args()
    from quant_fund_agent.comparison import rolling

    tickers = _resolve_tickers(args.data_dir, args.tickers)
    if not tickers:
        raise SystemExit(f"No tickers with bin*.csv found under {args.data_dir!r}.")
    preruns = ([s.strip() for s in args.preruns.split(",") if s.strip()]
               if args.preruns else list(rolling.DEFAULT_PRERUNS))
    name = args.name or f"rolling_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir = Path(args.out_root) / name

    log.info("rolling comparison '%s' → %s", name, batch_dir)
    log.info("  tickers : %s", tickers)
    log.info("  preruns : %s", preruns)
    log.info("  window  : %d IS + %d OOS, step %d  | models=%s  max_bars=%s  jobs=%d",
             args.is_len, args.oos_len, args.step, args.models or "all",
             args.max_bars if args.max_bars is not None else "full", args.jobs)

    result = rolling.run_rolling(
        tickers=tickers, data_dir=args.data_dir, batch_dir=batch_dir, preruns=preruns,
        is_len=args.is_len, oos_len=args.oos_len, step=args.step,
        models=args.models, max_bars=args.max_bars,
        importance_top_n=args.importance_top_n, jobs=args.jobs,
        force=args.force, aggregate_only=args.aggregate_only,
    )

    # Cross-ticker figures (OOS Sharpe / IC / Sharpe-ratio over time + Sharpe-vs-volume).
    try:
        fig_paths = build_cross_ticker_figures(batch_dir, args.data_dir)
    except Exception:  # noqa: BLE001 — a plotting failure must not lose the sweep
        log.exception("cross-ticker figure generation failed")
        fig_paths = []

    print("\n" + "=" * 80)
    print(f"Rolling comparison '{name}' → {batch_dir}")
    if result["statuses"]:
        ok = sum(s["status"] == "ok" for s in result["statuses"])
        skip = sum(s["status"] == "skipped" for s in result["statuses"])
        fail = sum(s["status"] == "failed" for s in result["statuses"])
        print(f"  runs        : {ok} ok, {skip} skipped, {fail} failed")
    print(f"  combined    : {batch_dir / 'combined'}")
    print(f"  per-ticker  : {batch_dir / 'per_ticker'}")
    print(f"  cross-ticker: {batch_dir / 'cross_ticker'} ({len(fig_paths)} figures)")
    print(f"  summary     : {batch_dir / 'summary.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
