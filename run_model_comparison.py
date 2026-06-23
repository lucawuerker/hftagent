"""Compare the factor output of different *research* LLMs, end to end.

Given several named preruns (each a batch of factors mined by a chosen research
LLM via ``run_factor_research.py``), evaluate and compare their factor sets on
three axes and emit presentation-ready figures, tables, a Markdown report and a
notebook under ``data/comparisons/<id>/``:

1. **single-factor IC**  — raw cross-sectional rank-IC per factor (LLM-free);
2. **brute-force ML**     — factors → model catalog + ensemble, OOS (LLM-free);
3. **downstream agents**  — factors → Selector→Architect→Statistician→PM (uses LLM).

Everything runs on whatever data is present *now* (factors needing absent fields
are filtered, and reported) and re-runs unchanged once more data is downloaded.

Examples
--------
::

    # Compare two existing preruns on all three tracks (downstream uses LLM).
    ./venv/bin/python run_model_comparison.py --preruns gpt4omini,claude

    # Fully offline: factor-quality + brute-force only, every prerun on disk.
    QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py --all --no-downstream

    # Fast brute-force of a single model on a small universe (quick iteration).
    QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py \
      --preruns gpt4omini,gpt5.4mini --no-downstream \
      --models random_forest --fast --n-tickers 20

    # Or tune the speed knobs explicitly (30% of rows, all models, 25 tickers).
    QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py --all --no-downstream \
      --train-sample-frac 0.3 --n-tickers 25

    # Pick the exact underlyings and a calendar IS/OOS split (train on 3 months,
    # test on the next) instead of a row-count / tail-fraction split.
    QF_USE_MCP=0 ./venv/bin/python run_model_comparison.py --all --no-downstream \
      --tickers AAPL,MSFT,CORN \
      --train-months 2024-06:2024-08 --oos-months 2024-09

    # Create the preruns first (spends research LLM), then compare.
    ./venv/bin/python run_model_comparison.py --research \
      --prerun-spec gpt4omini=gpt-4o-mini \
      --prerun-spec claude=claude-3-5-sonnet-latest:anthropic \
      --target-factors 50
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_model_comparison")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preruns", default=None,
                   help="Comma-separated prerun names to compare.")
    g.add_argument("--all", action="store_true",
                   help="Compare every prerun under data/factors/preruns/.")
    p.add_argument("--models", default=None,
                   help="Comma-separated brute-force models to run, e.g. "
                        "'--models random_forest' or '--models ridge,lightgbm' "
                        "(default: all catalog models).")
    p.add_argument("--no-ensemble", action="store_true",
                   help="Skip the equal-weight ensemble in the brute-force track.")
    p.add_argument("--include-seeds", action="store_true",
                   help="Also expose the 88 seed alphas to every track (default: only "
                        "the researched factors).")
    p.add_argument("--no-ic", action="store_true", help="Skip the single-factor IC track.")
    p.add_argument("--no-analytics", action="store_true",
                   help="Skip the factor-analytics track (diversity + deflation/importance).")
    p.add_argument("--no-bruteforce", action="store_true", help="Skip the brute-force ML track.")
    p.add_argument("--no-downstream", action="store_true",
                   help="Skip the (LLM-spending) downstream-agent track.")
    p.add_argument("--n-strategies", type=int, default=3,
                   help="Strategies built per prerun in the downstream track.")
    p.add_argument("--horizon", type=int, default=6,
                   help="Forecast horizon (bars) for brute-force + downstream.")
    p.add_argument("--oos-ratio", type=float, default=0.2,
                   help="Held-out tail fraction for the IS/OOS split (used unless "
                        "--train-months/--oos-months give a calendar split instead).")
    p.add_argument("--n-tickers", type=int, default=None,
                   help="How many underlyings to use (caps the universe). Fewer tickers "
                        "→ a smaller panel that speeds up every model AND the IC track. "
                        "Default = all tickers in the data dir.")
    p.add_argument("--tickers", default=None,
                   help="Explicit comma-separated underlyings to use (e.g. 'AAPL,MSFT,CORN'); "
                        "loads exactly these and OVERRIDES --n-tickers. Default = all tickers.")
    # ── calendar IS/OOS split (replaces the --oos-ratio tail split when both given) ──
    p.add_argument("--train-months", default=None, metavar="SPEC",
                   help="Calendar IS/train window instead of the --oos-ratio tail: a comma "
                        "list of months/dates ('2024-06,2024-07' or '2024-06-15') or an "
                        "inclusive range ('2024-06:2024-08'). Requires --oos-months; the "
                        "panel is restricted to train∪OOS so every track scores those months.")
    p.add_argument("--oos-months", default=None, metavar="SPEC",
                   help="Calendar OOS window (same format as --train-months); must be "
                        "disjoint from --train-months.")
    # ── speed knobs (brute-force model training) ──
    p.add_argument("--fast", action="store_true",
                   help="Fast preset: subsample training rows (--train-sample-frac "
                        "defaults to 0.1) and use lighter tree/boosting hyper-parameters. "
                        "Combine with --n-tickers for the biggest speedup.")
    p.add_argument("--train-sample-frac", type=float, default=None,
                   help="Fraction (0–1] of training ROWS used to FIT each model; the "
                        "backtest still uses all data. The main lever for the heavy "
                        "tree/boosting models. Default 1.0 (or 0.1 under --fast).")
    p.add_argument("--max-bars", type=int, default=None,
                   help="Uniformly subsample the panel to at most N timestamps (the biggest "
                        "speed lever on the intraday panel; slims EVERY track and removes the "
                        "brute-force OOM). Default = all bars (or 20000 under --fast).")
    p.add_argument("--analytics-max-rows", type=int, default=None,
                   help="Cap (timestamp×ticker) rows used in the analytics correlation / "
                        "importance fits (default 50000).")
    p.add_argument("--corr-threshold", type=float, default=None,
                   help="|corr| ≥ τ groups factors into a redundancy cluster (default 0.7).")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable persisting tables/figures after each track (crash-safety).")
    # ── ML-combined-signal vectorised backtest (brute-force track) ──
    p.add_argument("--fit-scope", choices=["pooled", "per_underlying"], default=None,
                   help="Fit ONE model across all underlyings ('pooled', the default) or "
                        "a SEPARATE model per underlying ('per_underlying'). Pooled suits "
                        "homogeneous, data-light universes (e.g. yfinance S&P100 stocks); "
                        "per_underlying suits heterogeneous, data-rich ones (e.g. the "
                        "LOBSTER ETFs across sectors, ~2340 bars/day each).")
    p.add_argument("--fit-standardize", choices=["per_underlying", "cross_sectional"], default=None,
                   help="How factors are standardised before the ML fit (default per_underlying).")
    p.add_argument("--position-mode", choices=["threshold", "sign", "continuous"], default=None,
                   help="Map the combined signal to a position (default threshold band).")
    p.add_argument("--position-threshold", type=float, default=None,
                   help="±t (in z units) for the threshold band (default 1.0).")
    p.add_argument("--position-zscore", choices=["expanding", "full", "rolling", "none"], default=None,
                   help="Per-underlying z-score basis for the position signal (default expanding).")
    p.add_argument("--position-zscore-window", type=int, default=None,
                   help="Window for the 'rolling' position z-score basis (default 500).")
    p.add_argument("--aggregation", choices=["portfolio", "per_underlying"], default=None,
                   help="Combine per-underlying P&L into one book, or report per-underlying mean/std (default portfolio).")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--out-dir", default=None, help="Override the output folder.")
    # ── optional research stage (creates the preruns first) ──
    p.add_argument("--research", action="store_true",
                   help="Create the preruns first by running run_factor_research.py per spec.")
    p.add_argument("--prerun-spec", action="append", default=[], metavar="name=model[:provider]",
                   help="A prerun to mine when --research is set (repeatable).")
    p.add_argument("--target-factors", type=int, default=50,
                   help="Factors to mine per prerun when --research is set.")
    p.add_argument("--dedup-scope", choices=["package", "prerun"], default="prerun",
                   help="De-dup scope for --research (default 'prerun' for a fair A/B).")
    return p.parse_args()


def _run_research(specs: list[str], target_factors: int, dedup_scope: str,
                  data_dir: str, n_tickers: int | None) -> list[str]:
    """Mine each ``name=model[:provider]`` prerun via run_factor_research.py (one process each)."""
    names: list[str] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Bad --prerun-spec {spec!r}; use name=model[:provider].")
        name, model = spec.split("=", 1)
        provider = None
        if ":" in model:
            model, provider = model.split(":", 1)
        cmd = [sys.executable, "run_factor_research.py", "--name", name, "--model", model,
               "--target-factors", str(target_factors), "--dedup-scope", dedup_scope,
               "--data-dir", data_dir]
        if provider:
            cmd += ["--llm-provider", provider]
        if n_tickers is not None:
            cmd += ["--n-tickers", str(n_tickers)]
        log.info("── research prerun '%s' (model=%s%s) ──", name, model,
                 f", provider={provider}" if provider else "")
        subprocess.run(cmd, check=True)
        names.append(name)
    return names


def _write_status(cfg, status: dict) -> None:
    """Write ``status.json`` (which tracks have completed) into the output dir."""
    import json

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(status, indent=2, default=str))


def _checkpoint(cfg, results: dict, status: dict) -> None:
    """Persist tables + figures so far so a later crash can't wipe progress.

    Both ``write_tables`` and ``render_figures`` overwrite, so calling them after
    every track is idempotent and cheap relative to the tracks themselves.
    """
    from quant_fund_agent.comparison import report

    if not cfg.checkpoint:
        return
    try:
        report.write_tables(cfg, results)
        report.render_figures(cfg, results)
        _write_status(cfg, status)
    except Exception as e:  # noqa: BLE001 — a checkpoint failure must not abort the run
        log.warning("checkpoint failed (continuing): %s", e)


def _run_track(name: str, fn, cfg, results: dict, status: dict) -> None:
    """Run one track, record status, and checkpoint — never aborting the whole run."""
    import time

    log.info("── track '%s' starting ──", name)
    t0 = time.time()
    try:
        fn()
        status["tracks"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 — one track failing must not lose the others
        log.exception("track '%s' failed: %s", name, e)
        status["tracks"][name] = {"status": "failed", "error": str(e)[:300],
                                  "seconds": round(time.time() - t0, 1)}
    _checkpoint(cfg, results, status)


def main() -> None:
    args = _parse_args()
    from quant_fund_agent.comparison import (
        analytics,
        bruteforce,
        downstream,
        factors,
        ic,
        report,
    )
    from quant_fund_agent.comparison.config import ComparisonConfig
    from quant_fund_agent.factors import preruns

    # ── optional: mine the preruns first ──
    researched: list[str] = []
    if args.research:
        researched = _run_research(args.prerun_spec, args.target_factors,
                                   args.dedup_scope, args.data_dir, args.n_tickers)

    prerun_names = (
        [s.strip() for s in args.preruns.split(",") if s.strip()] if args.preruns
        else researched or preruns.list_preruns()
    )
    if not prerun_names:
        raise SystemExit("No preruns to compare. Create some with run_factor_research.py "
                         "--name <id> --model <llm>, or pass --research --prerun-spec ….")

    # --fast implies a low default training-row fraction AND a slim panel unless
    # one is given (so "fast" is genuinely fast end-to-end on the intraday panel).
    train_sample_frac = (
        args.train_sample_frac if args.train_sample_frac is not None
        else (0.1 if args.fast else 1.0)
    )
    max_bars = (
        args.max_bars if args.max_bars is not None
        else (20_000 if args.fast else None)
    )
    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers else None
    )
    if tickers and args.n_tickers is not None:
        log.info("--tickers given (%d names) → ignoring --n-tickers=%d.", len(tickers), args.n_tickers)
    if bool(args.train_months) != bool(args.oos_months):
        raise SystemExit("--train-months and --oos-months must be given together "
                         "(a calendar split needs both a train and an OOS window).")

    cfg = ComparisonConfig(
        preruns=prerun_names,
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        include_ensemble=not args.no_ensemble,
        include_seeds=args.include_seeds,
        run_ic=not args.no_ic, run_analytics=not args.no_analytics,
        run_bruteforce=not args.no_bruteforce,
        run_downstream=not args.no_downstream,
        target_horizon=args.horizon, ic_horizons=(1, args.horizon, 60),
        oos_split_ratio=args.oos_ratio, n_strategies=args.n_strategies,
        data_dir=args.data_dir, n_tickers=args.n_tickers, max_bars=max_bars,
        tickers=tickers, is_window=args.train_months, oos_window=args.oos_months,
        fast=args.fast, train_sample_frac=train_sample_frac,
        checkpoint=not args.no_checkpoint,
        **({"analytics_max_rows": args.analytics_max_rows} if args.analytics_max_rows else {}),
        **({"corr_threshold": args.corr_threshold} if args.corr_threshold is not None else {}),
        **({"fit_scope": args.fit_scope} if args.fit_scope else {}),
        **({"fit_standardize": args.fit_standardize} if args.fit_standardize else {}),
        **({"position_mode": args.position_mode} if args.position_mode else {}),
        **({"position_threshold": args.position_threshold} if args.position_threshold is not None else {}),
        **({"position_zscore_basis": args.position_zscore} if args.position_zscore else {}),
        **({"position_zscore_window": args.position_zscore_window} if args.position_zscore_window else {}),
        **({"backtest_aggregation": args.aggregation} if args.aggregation else {}),
    )
    if args.out_dir:
        cfg.output_root = str(Path(args.out_dir).parent)
        cfg.comparison_id = Path(args.out_dir).name

    log.info("Comparison '%s' over %d prerun(s): %s", cfg.comparison_id,
             len(prerun_names), prerun_names)
    if cfg.fast or cfg.train_sample_frac < 1.0 or cfg.n_tickers is not None or cfg.max_bars:
        log.info("speed: fast=%s  train_sample_frac=%.3g  n_tickers=%s  max_bars=%s  models=%s",
                 cfg.fast, cfg.train_sample_frac, cfg.n_tickers, cfg.max_bars,
                 cfg.models or "all")
    if cfg.tickers:
        log.info("universe: explicit tickers %s", cfg.tickers)

    # ── shared panel + per-prerun usable factor ids ──
    panel = factors.load_panel_cached(
        cfg.data_dir, cfg.n_tickers, cfg.max_bars,
        tickers=cfg.tickers, is_window=cfg.is_window, oos_window=cfg.oos_window)
    universe = next(iter(panel.values())).shape[1] if panel else 0
    if cfg.is_window or cfg.oos_window:
        idx = next(iter(panel.values())).index if panel else None
        try:
            im, om = cfg.split_masks(idx)
        except ValueError as e:
            raise SystemExit(str(e))
        log.info("calendar IS/OOS split: IS=%d bars [%s], OOS=%d bars [%s]",
                 int(im.sum()), cfg.is_window, int(om.sum()), cfg.oos_window)
        if idx is not None and (im.sum() == 0 or om.sum() == 0):
            log.warning("a calendar window selected 0 bars — check the months lie within the "
                        "loaded data range (%s … %s).", idx.min(), idx.max())
    if universe < 2:
        log.warning("universe has %d ticker(s): this comparison is CROSS-SECTIONAL "
                    "(rank-IC, the ML fits and factor correlations all need ≥2 names), so "
                    "IC/brute-force will be degenerate. For speed prefer --max-bars over "
                    "--n-tickers.", universe)
    names_map = factors.factor_names(prerun_names)
    prerun_models = {p: (preruns.read_manifest(p).get("llm_model", "?")) for p in prerun_names}

    usable: dict[str, list[str]] = {}
    usability: dict[str, dict] = {}
    for p in prerun_names:
        all_ids = factors.prerun_factor_ids(p, include_seeds=cfg.include_seeds)
        ok, dropped = factors.usable_factor_ids(all_ids, panel)
        usable[p] = ok
        usability[p] = {"n_total": len(all_ids), "n_usable": len(ok),
                        "n_dropped": len(dropped), "dropped": dropped}
        log.info("prerun '%s': %d/%d factors usable on current data",
                 p, len(ok), len(all_ids))

    results: dict = {"usability": usability, "prerun_models": prerun_models}
    status: dict = {"comparison_id": cfg.comparison_id, "tracks": {}}
    _checkpoint(cfg, results, status)  # persist usability immediately

    # ── track 1: single-factor IC ──
    if cfg.run_ic:
        def _ic() -> None:
            ic_rows: list = []
            for p in prerun_names:
                ic_rows += ic.evaluate_prerun_ic(p, usable[p], panel,
                                                 tuple(cfg.ic_horizons), names_map)
            results["ic_rows"] = ic_rows
            results["ic_summary"] = ic.summarise_ic(ic_rows, tuple(cfg.ic_horizons))
        _run_track("ic", _ic, cfg, results, status)

    # ── track 2: factor analytics (diversity/redundancy + deflation/importance) ──
    if cfg.run_analytics:
        def _an() -> None:
            div_summary, div_factor, corr_mats = [], [], {}
            mv_summary, imp_rows = [], []
            for p in prerun_names:
                if not usable[p]:
                    continue
                s, f, corr = analytics.evaluate_prerun_diversity(p, usable[p], panel, cfg, names_map)
                div_summary.append(s); div_factor += f; corr_mats[p] = corr
                ms, ir = analytics.evaluate_prerun_modelview(
                    p, usable[p], panel, cfg, ic_rows=results.get("ic_rows"), names=names_map)
                mv_summary.append(ms); imp_rows += ir
            results["diversity_summary"] = div_summary
            results["diversity_factor"] = div_factor
            results["corr_matrices"] = corr_mats
            results["modelview_summary"] = mv_summary
            results["importance_rows"] = imp_rows
        _run_track("analytics", _an, cfg, results, status)

    # ── track 3: brute-force ML ──
    if cfg.run_bruteforce:
        def _bf() -> None:
            bf_rows: list = []
            for p in prerun_names:
                if not usable[p]:
                    log.warning("prerun '%s' has no usable factors — skipping brute-force.", p)
                    continue
                bf_rows += bruteforce.evaluate_prerun_models(p, usable[p], panel, cfg)
            results["bruteforce_rows"] = bf_rows
        _run_track("bruteforce", _bf, cfg, results, status)

    # ── track 4: downstream agents (LLM) ──
    if cfg.run_downstream:
        def _ds() -> None:
            ds_summary: list = []
            ds_strategies: list = []
            for p in prerun_names:
                summary, rows = downstream.evaluate_prerun_downstream(p, cfg)
                ds_summary.append(summary)
                ds_strategies += rows
            results["downstream_summary"] = ds_summary
            results["downstream_strategies"] = ds_strategies
        _run_track("downstream", _ds, cfg, results, status)

    # ── final persist: tables, figures, report, notebook ──
    # Tables were already checkpointed after each track, so a late figure/report
    # error must not abort with a traceback — degrade to what we have on disk.
    tables = report.write_tables(cfg, results)
    figs: dict = {}
    report_md = cfg.output_dir / "report.md"
    nb = cfg.output_dir / "comparison.ipynb"
    try:
        figs = report.render_figures(cfg, results)
        report_md = report.write_report_md(cfg, results, figs)
        nb = report.build_comparison_notebook(cfg)
    except Exception as e:  # noqa: BLE001 — tables are safe; report is best-effort
        log.exception("final figures/report failed (tables are on disk): %s", e)
    _write_status(cfg, status)

    print("\n" + "=" * 80)
    print(f"Comparison '{cfg.comparison_id}' → {cfg.output_dir}")
    print(f"  preruns         : {prerun_names}")
    print(f"  universe        : {cfg.tickers if cfg.tickers else f'{universe} tickers'}")
    if cfg.is_window or cfg.oos_window:
        print(f"  split           : IS=[{cfg.is_window}]  OOS=[{cfg.oos_window}]")
    print(f"  tracks          : "
          f"{'IC ' if cfg.run_ic else ''}{'analytics ' if cfg.run_analytics else ''}"
          f"{'brute-force ' if cfg.run_bruteforce else ''}"
          f"{'downstream' if cfg.run_downstream else ''}".strip())
    print(f"  status          : {status['tracks']}")
    print(f"  figures         : {len(figs)} → {cfg.figures_dir}")
    print(f"  tables          : {len(tables)} CSV/JSON")
    print(f"  report          : {report_md}")
    print(f"  notebook        : {nb}")
    print("=" * 80)


if __name__ == "__main__":
    main()
