"""Compute IC / ICIR for registered factors and persist to FactorDatabase JSON.

Usage::

    # single factor (quick proof-of-concept)
    python run_all_factors.py --factor alpha_101

    # all factors (overnight run)
    python run_all_factors.py

    # custom options
    python run_all_factors.py --factor alpha_007 --horizon 12 --data-dir ticker_data
"""

from __future__ import annotations

import argparse
import logging
import time
import traceback

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_all_factors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest registered factors")
    parser.add_argument("--factor", type=str, default=None,
                        help="Run a single factor by its factor_id (e.g. alpha_101). "
                             "Omit to run all factors.")
    parser.add_argument("--data-dir", default="ticker_data", help="Path to ticker_data/")
    parser.add_argument("--horizon", type=int, default=6,
                        help="Default horizon used for top-level IC fields (legacy compatibility).")
    parser.add_argument("--horizons", type=str, default="1,6,60",
                        help="Comma-separated horizons in bars for multi-horizon IC (e.g. 1,6,60).")
    parser.add_argument("--output", default="data/factors/factor_db.json", help="Output JSON path")
    args = parser.parse_args()

    from quant_fund_agent.backtesting.data_loader import load_panel
    from quant_fund_agent.backtesting.engine import backtest_factor
    from quant_fund_agent.databases.factor_db import FactorDatabase
    from quant_fund_agent.factors import discover_factors, get_all_factor_classes
    from quant_fund_agent.schemas import BacktestMetrics, FactorRecord, FactorStatus

    discover_factors()
    all_classes = get_all_factor_classes()
    log.info("Discovered %d factor classes", len(all_classes))
    horizons = tuple(sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()}))
    log.info("Using horizons (bars): %s", horizons)

    if args.factor:
        if args.factor not in all_classes:
            log.error("Factor %r not found. Available: %s",
                      args.factor, ", ".join(sorted(all_classes)))
            return
        run_classes = {args.factor: all_classes[args.factor]}
    else:
        run_classes = dict(sorted(all_classes.items()))

    log.info("Will evaluate %d factor(s)", len(run_classes))

    log.info("Loading panel data from %s …", args.data_dir)
    t0 = time.time()
    data = load_panel(args.data_dir)
    n_tickers = data["close"].shape[1]
    n_bars = data["close"].shape[0]
    log.info("Loaded %d tickers × %d bars (%.1fs)", n_tickers, n_bars, time.time() - t0)

    db = FactorDatabase()
    results_rows: list[dict] = []

    for i, (fid, cls) in enumerate(run_classes.items(), 1):
        factor = cls()
        log.info("[%d/%d] %s (%s) …", i, len(run_classes), fid, factor.category)
        t1 = time.time()

        try:
            metrics = backtest_factor(
                factor,
                data,
                horizon=args.horizon,
                horizons=horizons,
            )
        except Exception:
            log.warning("  FAILED: %s", traceback.format_exc().splitlines()[-1])
            metrics = BacktestMetrics(extra={"error": traceback.format_exc()})

        elapsed = time.time() - t1
        record = FactorRecord(
            id=factor.factor_id,
            name=factor.name,
            class_name=cls.__name__,
            description=factor.description,
            category=factor.category,
            status=FactorStatus.BACKTESTED,
            backtest_metrics=metrics,
        )
        db.add_factor(record)

        row = {
            "factor_id": fid,
            "name": factor.name,
            "category": factor.category,
            "IC": metrics.information_coefficient,
            "IC_std": metrics.ic_std,
            "ICIR": metrics.ic_ir,
            "IC_hit": metrics.ic_hit_rate,
            "n_ts": metrics.extra.get("n_timestamps"),
            "IC_1": (metrics.ic_by_horizon.get("1") or {}).get("ic"),
            "ICIR_1": (metrics.ic_by_horizon.get("1") or {}).get("ic_ir"),
            "IC_6": (metrics.ic_by_horizon.get("6") or {}).get("ic"),
            "ICIR_6": (metrics.ic_by_horizon.get("6") or {}).get("ic_ir"),
            "IC_60": (metrics.ic_by_horizon.get("60") or {}).get("ic"),
            "ICIR_60": (metrics.ic_by_horizon.get("60") or {}).get("ic_ir"),
            "time_s": round(elapsed, 1),
        }
        results_rows.append(row)
        log.info("  IC=%.4f  ICIR=%.4f  (%.1fs)",
                 metrics.information_coefficient or 0,
                 metrics.ic_ir or 0,
                 elapsed)

    db.save_to_json(args.output)
    log.info("Saved factor database to %s", args.output)

    df = pd.DataFrame(results_rows)
    df = df.sort_values("ICIR", ascending=False, key=lambda s: s.abs(), na_position="last")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n" + "=" * 100)
    print("FACTOR IC SUMMARY  (sorted by |ICIR|)")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nTotal factors evaluated: {len(df)}")
    if len(df) > 1:
        print(f"Factors with IC > 0:     {(df['IC'] > 0).sum()}")
        print(f"Factors with |ICIR|>0.1: {(df['ICIR'].abs() > 0.1).sum()}")


if __name__ == "__main__":
    main()
