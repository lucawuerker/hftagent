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
    p.add_argument("--no-bruteforce", action="store_true", help="Skip the brute-force ML track.")
    p.add_argument("--no-downstream", action="store_true",
                   help="Skip the (LLM-spending) downstream-agent track.")
    p.add_argument("--n-strategies", type=int, default=3,
                   help="Strategies built per prerun in the downstream track.")
    p.add_argument("--horizon", type=int, default=6,
                   help="Forecast horizon (bars) for brute-force + downstream.")
    p.add_argument("--oos-ratio", type=float, default=0.2)
    p.add_argument("--n-tickers", type=int, default=None,
                   help="How many underlyings to use (caps the universe). Fewer tickers "
                        "→ a smaller panel that speeds up every model AND the IC track. "
                        "Default = all tickers in the data dir.")
    # ── speed knobs (brute-force model training) ──
    p.add_argument("--fast", action="store_true",
                   help="Fast preset: subsample training rows (--train-sample-frac "
                        "defaults to 0.1) and use lighter tree/boosting hyper-parameters. "
                        "Combine with --n-tickers for the biggest speedup.")
    p.add_argument("--train-sample-frac", type=float, default=None,
                   help="Fraction (0–1] of training ROWS used to FIT each model; the "
                        "backtest still uses all data. The main lever for the heavy "
                        "tree/boosting models. Default 1.0 (or 0.1 under --fast).")
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


def main() -> None:
    args = _parse_args()
    from quant_fund_agent.comparison import bruteforce, downstream, factors, ic, report
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

    # --fast implies a low default training-row fraction unless one is given.
    train_sample_frac = (
        args.train_sample_frac if args.train_sample_frac is not None
        else (0.1 if args.fast else 1.0)
    )

    cfg = ComparisonConfig(
        preruns=prerun_names,
        models=[m.strip() for m in args.models.split(",")] if args.models else None,
        include_ensemble=not args.no_ensemble,
        include_seeds=args.include_seeds,
        run_ic=not args.no_ic, run_bruteforce=not args.no_bruteforce,
        run_downstream=not args.no_downstream,
        target_horizon=args.horizon, ic_horizons=(1, args.horizon, 60),
        oos_split_ratio=args.oos_ratio, n_strategies=args.n_strategies,
        data_dir=args.data_dir, n_tickers=args.n_tickers,
        fast=args.fast, train_sample_frac=train_sample_frac,
    )
    if args.out_dir:
        cfg.output_root = str(Path(args.out_dir).parent)
        cfg.comparison_id = Path(args.out_dir).name

    log.info("Comparison '%s' over %d prerun(s): %s", cfg.comparison_id,
             len(prerun_names), prerun_names)
    if cfg.fast or cfg.train_sample_frac < 1.0 or cfg.n_tickers is not None:
        log.info("speed: fast=%s  train_sample_frac=%.3g  n_tickers=%s  models=%s",
                 cfg.fast, cfg.train_sample_frac, cfg.n_tickers,
                 cfg.models or "all")

    # ── shared panel + per-prerun usable factor ids ──
    panel = factors.load_panel_cached(cfg.data_dir, cfg.n_tickers)
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

    # ── track 1: single-factor IC ──
    if cfg.run_ic:
        ic_rows: list = []
        for p in prerun_names:
            ic_rows += ic.evaluate_prerun_ic(p, usable[p], panel,
                                             tuple(cfg.ic_horizons), names_map)
        results["ic_rows"] = ic_rows
        results["ic_summary"] = ic.summarise_ic(ic_rows, tuple(cfg.ic_horizons))

    # ── track 2: brute-force ML ──
    if cfg.run_bruteforce:
        bf_rows: list = []
        for p in prerun_names:
            if not usable[p]:
                log.warning("prerun '%s' has no usable factors — skipping brute-force.", p)
                continue
            bf_rows += bruteforce.evaluate_prerun_models(p, usable[p], cfg)
        results["bruteforce_rows"] = bf_rows

    # ── track 3: downstream agents (LLM) ──
    if cfg.run_downstream:
        ds_summary: list = []
        ds_strategies: list = []
        for p in prerun_names:
            summary, rows = downstream.evaluate_prerun_downstream(p, cfg)
            ds_summary.append(summary)
            ds_strategies += rows
        results["downstream_summary"] = ds_summary
        results["downstream_strategies"] = ds_strategies

    # ── persist tables, figures, report, notebook ──
    tables = report.write_tables(cfg, results)
    figs = report.render_figures(cfg, results)
    report_md = report.write_report_md(cfg, results, figs)
    nb = report.build_comparison_notebook(cfg)

    print("\n" + "=" * 80)
    print(f"Comparison '{cfg.comparison_id}' → {cfg.output_dir}")
    print(f"  preruns         : {prerun_names}")
    print(f"  tracks          : "
          f"{'IC ' if cfg.run_ic else ''}{'brute-force ' if cfg.run_bruteforce else ''}"
          f"{'downstream' if cfg.run_downstream else ''}".strip())
    print(f"  figures         : {len(figs)} → {cfg.figures_dir}")
    print(f"  tables          : {len(tables)} CSV/JSON")
    print(f"  report          : {report_md}")
    print(f"  notebook        : {nb}")
    print("=" * 80)


if __name__ == "__main__":
    main()
