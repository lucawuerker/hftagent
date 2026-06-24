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

    print("\n" + "=" * 80)
    print(f"Rolling comparison '{name}' → {batch_dir}")
    if result["statuses"]:
        ok = sum(s["status"] == "ok" for s in result["statuses"])
        skip = sum(s["status"] == "skipped" for s in result["statuses"])
        fail = sum(s["status"] == "failed" for s in result["statuses"])
        print(f"  runs        : {ok} ok, {skip} skipped, {fail} failed")
    print(f"  combined    : {batch_dir / 'combined'}")
    print(f"  per-ticker  : {batch_dir / 'per_ticker'}")
    print(f"  summary     : {batch_dir / 'summary.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
