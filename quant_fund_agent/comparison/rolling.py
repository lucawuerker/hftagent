"""Rolling-window factor-set comparison: driver + aggregator (LLM-free).

For **every ticker** under ``ticker_data/`` this drives
``run_model_comparison.py`` over a **rolling IS/OOS month window** — 2 months
in-sample + the next month out-of-sample by default, stepping one month forward
each time (the prior OOS month becomes the second IS month) — and then
**aggregates** every run's CSVs into combined tables and a per-ticker
*feature-importance-over-OOS-months* analysis.

Why a subprocess per (ticker, window)?  The LOBSTER intraday panel is large; the
single robust way to guarantee no OOM across ~280 runs is to run each window in
its own process, so the OS reclaims all of its memory before the next one starts.
Each child loads **one** ticker and restricts to just the 3 window months.  The
sweep is **resumable** (a window whose ``status.json`` shows every track ``ok`` is
skipped) and **robust** (a failed run is logged and the sweep continues).

The comparison is run **per underlying, never cross-sectional** (``--fit-scope
per_underlying --fit-standardize per_underlying``), so the IC, analytics and
brute-force tracks are all meaningful for a single ticker.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("comparison.rolling")

# bin{YYYYMM}.csv → "YYYY-MM"
_BIN_RE = re.compile(r"bin(\d{4})(\d{2})\.csv$")

DEFAULT_PRERUNS = ["gpt4omini120650", "gpt5.4mini120650", "main"]
# Default LOBSTER data flags (the active quant.config.yaml is yfinance, so the
# rolling sweep overrides the provider explicitly).
DEFAULT_DATA_FLAGS = ["--provider", "lobster", "--asset-class", "equity",
                      "--frequency", "10s"]


# ── window generation ────────────────────────────────────────────────────────

def discover_months(ticker_dir: Path) -> list[str]:
    """Sorted ``"YYYY-MM"`` months present as ``bin{YYYYMM}.csv`` under a ticker dir."""
    months: set[str] = set()
    for p in Path(ticker_dir).glob("bin*.csv"):
        m = _BIN_RE.search(p.name)
        if m:
            months.add(f"{m.group(1)}-{m.group(2)}")
    return sorted(months)


@dataclass(frozen=True)
class Window:
    """One rolling IS/OOS month window."""

    is_months: tuple[str, ...]
    oos_months: tuple[str, ...]

    @property
    def label(self) -> str:
        """Folder/label key — the OOS month(s) (chronologically sortable)."""
        return "_".join(self.oos_months)

    @property
    def is_spec(self) -> str:
        return ",".join(self.is_months)

    @property
    def oos_spec(self) -> str:
        return ",".join(self.oos_months)


def build_windows(months: list[str], is_len: int = 2, oos_len: int = 1,
                  step: int = 1) -> list[Window]:
    """Rolling windows over ``months``: ``is_len`` IS months then ``oos_len`` OOS.

    Steps ``step`` months forward; with the defaults the previous OOS month becomes
    the second IS month.  Returns ``[]`` if there aren't enough months for one
    full window.
    """
    out: list[Window] = []
    span = is_len + oos_len
    for i in range(0, len(months) - span + 1, step):
        out.append(Window(
            tuple(months[i:i + is_len]),
            tuple(months[i + is_len:i + span]),
        ))
    return out


# ── running one window in an isolated subprocess ──────────────────────────────

def _completed(out_dir: Path) -> bool:
    """True if ``out_dir/status.json`` shows every track completed ``ok``."""
    sp = out_dir / "status.json"
    if not sp.exists():
        return False
    try:
        st = json.loads(sp.read_text())
    except Exception:  # noqa: BLE001 — a corrupt status forces a re-run
        return False
    tracks = st.get("tracks", {})
    return bool(tracks) and all(t.get("status") == "ok" for t in tracks.values())


def run_window(
    ticker: str, window: Window, batch_dir: Path, preruns: list[str],
    data_dir: str, *, models: str | None, max_bars: int | None,
    importance_top_n: int, force: bool, python: str = sys.executable,
) -> dict[str, Any]:
    """Run ``run_model_comparison.py`` for one (ticker, window) in a child process."""
    out_dir = batch_dir / "runs" / ticker / window.label
    rec = {"ticker": ticker, "oos_month": window.label, "is_window": window.is_spec,
           "out_dir": str(out_dir)}
    if not force and _completed(out_dir):
        log.info("· skip %s / %s (already complete)", ticker, window.label)
        return {**rec, "status": "skipped", "returncode": 0}

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "run_model_comparison.py",
           "--preruns", ",".join(preruns),
           "--tickers", ticker,
           "--data-dir", data_dir,
           "--train-months", window.is_spec,
           "--oos-months", window.oos_spec,
           "--no-downstream",
           "--fit-scope", "per_underlying",
           "--fit-standardize", "per_underlying",
           "--importance-top-n", str(importance_top_n),
           "--out-dir", str(out_dir)]
    cmd += DEFAULT_DATA_FLAGS
    if models:
        cmd += ["--models", models]
    if max_bars is not None:
        cmd += ["--max-bars", str(max_bars)]

    env = dict(os.environ)
    env["QF_USE_MCP"] = "0"  # LLM-free tracks need no MCP modelling server
    log.info("▶ %s / IS=[%s] OOS=[%s] …", ticker, window.is_spec, window.oos_spec)
    log_path = out_dir / "run.log"
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, check=False)
    ok = proc.returncode == 0 and _completed(out_dir)
    if not ok:
        log.warning("✗ %s / %s failed (rc=%s) — see %s", ticker, window.label,
                    proc.returncode, log_path)
    return {**rec, "status": "ok" if ok else "failed", "returncode": proc.returncode}


# ── top-level driver ──────────────────────────────────────────────────────────

def run_rolling(
    *, tickers: list[str], data_dir: str, batch_dir: Path, preruns: list[str],
    is_len: int, oos_len: int, step: int, models: str | None, max_bars: int | None,
    importance_top_n: int, jobs: int, force: bool, aggregate_only: bool,
) -> dict[str, Any]:
    """Drive the whole sweep (unless ``aggregate_only``) then aggregate the results."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, Any]] = []

    if not aggregate_only:
        jobs_q: list[tuple[str, Window]] = []
        for ticker in tickers:
            months = discover_months(Path(data_dir) / ticker)
            wins = build_windows(months, is_len, oos_len, step)
            if not wins:
                log.warning("ticker %s: only %d months — no %d+%d window fits; skipping",
                            ticker, len(months), is_len, oos_len)
                continue
            log.info("ticker %s: %d months → %d windows", ticker, len(months), len(wins))
            jobs_q += [(ticker, w) for w in wins]

        log.info("rolling sweep: %d (ticker,window) runs over %d tickers (jobs=%d)",
                 len(jobs_q), len(tickers), jobs)

        def _do(item: tuple[str, Window]) -> dict[str, Any]:
            t, w = item
            return run_window(t, w, batch_dir, preruns, data_dir, models=models,
                              max_bars=max_bars, importance_top_n=importance_top_n,
                              force=force)

        if jobs <= 1:
            for i, item in enumerate(jobs_q, 1):
                statuses.append(_do(item))
                _write_manifest(batch_dir, statuses, preruns, is_len, oos_len, step)
                log.info("  [%d/%d] done", i, len(jobs_q))
        else:
            from concurrent.futures import ThreadPoolExecutor  # subprocess releases GIL
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for st in ex.map(_do, jobs_q):
                    statuses.append(st)
            _write_manifest(batch_dir, statuses, preruns, is_len, oos_len, step)

        ok = sum(s["status"] == "ok" for s in statuses)
        skip = sum(s["status"] == "skipped" for s in statuses)
        fail = sum(s["status"] == "failed" for s in statuses)
        log.info("sweep complete: %d ok, %d skipped, %d failed", ok, skip, fail)

    paths = aggregate(batch_dir)
    return {"statuses": statuses, "aggregate": {k: str(v) for k, v in paths.items()}}


def _write_manifest(batch_dir: Path, statuses: list[dict[str, Any]], preruns: list[str],
                    is_len: int, oos_len: int, step: int) -> None:
    (batch_dir / "manifest.json").write_text(json.dumps({
        "preruns": preruns, "is_len": is_len, "oos_len": oos_len, "step": step,
        "runs": statuses,
    }, indent=2, default=str))


# ── aggregation ───────────────────────────────────────────────────────────────

def _read_csv(p: Path):
    import pandas as pd

    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception:  # noqa: BLE001 — a truncated CSV from a crashed run
        return pd.DataFrame()


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _iter_runs(batch_dir: Path):
    """Yield ``(ticker, oos_label, run_dir)`` for every completed-or-not run dir."""
    runs_root = batch_dir / "runs"
    if not runs_root.is_dir():
        return
    for tdir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for odir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            yield tdir.name, odir.name, odir


def aggregate(batch_dir: Path) -> dict[str, Path]:
    """Combine every run's CSVs + build the per-ticker importance-over-months views."""
    import pandas as pd

    combined = batch_dir / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list] = {"bruteforce": [], "importance": [], "diversity": [], "ic": []}
    files = {"bruteforce": "bruteforce_results.csv", "importance": "analytics_importance.csv",
             "diversity": "analytics_diversity_summary.csv", "ic": "ic_results.csv"}

    for ticker, oos, d in _iter_runs(batch_dir):
        meta = _read_json(d / "config.json")
        is_window = meta.get("is_window")
        for key, fname in files.items():
            df = _read_csv(d / fname)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "ticker", ticker)
            df.insert(1, "oos_month", oos)
            df.insert(2, "is_window", is_window)
            buckets[key].append(df)

    paths: dict[str, Path] = {}
    combined_frames: dict[str, Any] = {}
    for key, frames in buckets.items():
        if not frames:
            continue
        all_df = pd.concat(frames, ignore_index=True)
        fp = combined / f"{key}_all.csv"
        all_df.to_csv(fp, index=False)
        paths[f"{key}_all"] = fp
        combined_frames[key] = all_df
        log.info("combined %s: %d rows → %s", key, len(all_df), fp)

    paths.update(_per_ticker_analysis(
        batch_dir, combined_frames.get("importance"), combined_frames.get("bruteforce")))
    paths["summary"] = _write_summary(batch_dir, combined_frames)
    return paths


def _months_sorted(values) -> list[str]:
    return sorted({str(v) for v in values})


def _per_ticker_analysis(batch_dir: Path, imp_all, bf_all) -> dict[str, Path]:
    """Per ticker: factor×OOS-month importance matrices + heatmaps + perf lines."""
    paths: dict[str, Path] = {}
    if imp_all is None or imp_all.empty:
        return paths
    tickers = sorted(imp_all["ticker"].unique())
    for ticker in tickers:
        tdir = batch_dir / "per_ticker" / ticker
        tdir.mkdir(parents=True, exist_ok=True)
        sub = imp_all[imp_all["ticker"] == ticker]
        for (prerun, model), g in sub.groupby(["prerun", "model"]):
            pivot = g.pivot_table(index="name", columns="oos_month",
                                  values="importance", aggfunc="mean")
            pivot = pivot.reindex(columns=_months_sorted(pivot.columns))
            csv = tdir / f"importance_over_months__{prerun}__{model}.csv"
            pivot.to_csv(csv)
            paths[f"imp_{ticker}_{prerun}_{model}"] = csv
            fig = _importance_heatmap(pivot, tdir, ticker, str(prerun), str(model))
            if fig is not None:
                paths[f"impfig_{ticker}_{prerun}_{model}"] = fig
        if bf_all is not None and not bf_all.empty:
            for fp in _performance_over_months(bf_all[bf_all["ticker"] == ticker], tdir, ticker):
                paths[f"perf_{ticker}_{fp.stem}"] = fp
    return paths


def _importance_heatmap(pivot, out_dir: Path, ticker: str, prerun: str, model: str,
                        topk: int = 20):
    """Heatmap of the top-``topk`` factors (by mean importance) over the OOS months."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if pivot is None or pivot.empty:
        return None
    M = pivot.fillna(0.0)
    order = M.mean(axis=1).sort_values(ascending=False).index[:topk]
    M = M.loc[order]
    if M.empty or M.shape[1] == 0:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * M.shape[1] + 3),
                                    max(3, 0.34 * len(order) + 1.5)))
    im = ax.imshow(M.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7)
    ax.set_title(f"{ticker} — {prerun} / {model}\ntop-{len(order)} factor importance over OOS months",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    p = out_dir / f"importance_over_months__{prerun}__{model}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _performance_over_months(bf, out_dir: Path, ticker: str,
                             model_pref: tuple[str, ...] = ("ensemble",)) -> list[Path]:
    """Line plots of OOS Sharpe / OOS IC per prerun across the OOS months."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: list[Path] = []
    if bf is None or bf.empty or "model" not in bf:
        return out
    models = list(bf["model"].dropna().unique())
    if not models:
        return out
    model = next((m for m in model_pref if m in models), models[0])
    sub = bf[bf["model"] == model]
    for metric in ("oos_sharpe", "oos_ic"):
        if metric not in sub or sub[metric].notna().sum() == 0:
            continue
        piv = sub.pivot_table(index="oos_month", columns="prerun", values=metric, aggfunc="mean")
        piv = piv.reindex(_months_sorted(piv.index))
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(piv.index) + 3), 4))
        for col in piv.columns:
            ax.plot(piv.index, piv[col], marker="o", label=str(col))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{ticker} — {metric} over OOS months ({model})", fontsize=9)
        ax.set_xlabel("OOS month")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"performance_{metric}__{model}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        out.append(p)
    return out


def _write_summary(batch_dir: Path, frames: dict[str, Any]) -> Path:
    import pandas as pd

    lines = ["# Rolling-window factor-set comparison\n"]
    bf = frames.get("bruteforce")
    if bf is not None and not bf.empty:
        n_runs = bf[["ticker", "oos_month"]].drop_duplicates().shape[0]
        lines.append(f"- runs aggregated: **{n_runs}** (ticker × OOS-month)\n")
        lines.append(f"- tickers: {', '.join(sorted(bf['ticker'].unique()))}\n")
        if "oos_sharpe" in bf and bf["oos_sharpe"].notna().any():
            per = (bf.dropna(subset=["oos_sharpe"]).groupby("prerun")["oos_sharpe"].mean()
                   .sort_values(ascending=False))
            lines.append("\n## Mean OOS Sharpe across all runs & models, by factor set\n")
            lines.append(_md_series(per))
        if {"model", "oos_sharpe"} <= set(bf.columns):
            piv = (bf.dropna(subset=["oos_sharpe"])
                   .pivot_table(index="prerun", columns="model", values="oos_sharpe", aggfunc="mean"))
            if not piv.empty:
                lines.append("\n## Mean OOS Sharpe by factor set × model\n")
                lines.append(_md_frame(piv.round(3)))
    lines.append("\n## Outputs\n")
    lines.append("- `combined/` — `bruteforce_all.csv`, `importance_all.csv`, "
                 "`diversity_all.csv`, `ic_all.csv` (every run, tagged with "
                 "`ticker, oos_month, is_window`).\n")
    lines.append("- `per_ticker/<ticker>/` — `importance_over_months__<prerun>__<model>.csv` "
                 "(+ heatmap PNG) and `performance_<metric>__<model>.png`.\n")
    lines.append("- `runs/<ticker>/<oos_month>/` — each window's full comparison "
                 "(`report.md`, figures, `run.log`).\n")
    p = batch_dir / "summary.md"
    p.write_text("\n".join(lines))
    log.info("wrote %s", p)
    return p


def _md_series(s) -> str:
    return "\n".join(f"- `{k}` = {v:.4f}" for k, v in s.items()) + "\n"


def _md_frame(df) -> str:
    """Render a DataFrame as a GitHub-markdown table (no `tabulate` dependency)."""
    cols = ["prerun", *[str(c) for c in df.columns]]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + ["" if v != v else f"{v:.3f}" for v in row]  # v!=v → NaN
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows]) + "\n"
