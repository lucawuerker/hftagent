"""CLI for the landing-page example generator.

::

    # 1. Run N real pipeline attempts under a scope, dumping every candidate.
    python -m showcase_pipeline.landing_examples run \
        --config-name yfinance_equity_sp100 --prerun lodestar_demo \
        --n-strategies 8 --max-iterations 6 --no-seeds

    # 2. Inspect every dumped candidate's harness metrics.
    python -m showcase_pipeline.landing_examples list \
        --config-name yfinance_equity_sp100 --prerun lodestar_demo

    # 3. Export chosen candidates into the company-brain marketing repo.
    python -m showcase_pipeline.landing_examples export \
        --config-name yfinance_equity_sp100 --prerun lodestar_demo \
        --pick 1:robust-example 4:overfit-example \
        --out "../company-brain/marketing/examples"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("landing_examples")


def _resolve_scope(args):
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.workspace import Scope

    if args.config:
        os.environ["QF_CONFIG_FILE"] = args.config
    config_name = args.config_name or default_config_name(get_settings().data)
    return Scope(config_name, args.prerun)


def _cmd_run(args) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")
    if args.config:
        os.environ["QF_CONFIG_FILE"] = args.config

    from .driver import run_batch

    dumped = run_batch(
        config_name=args.config_name,
        prerun=args.prerun,
        n_strategies=args.n_strategies,
        max_iterations=args.max_iterations,
        oos_ratio=args.oos_ratio,
        no_seeds=args.no_seeds,
        n_tickers=args.n_tickers,
        target_horizon=args.target_horizon,
        position_construction=args.position_construction,
    )
    print(f"\nDumped {len(dumped)} attempt(s). Next: "
          f"python -m showcase_pipeline.landing_examples list "
          f"--config-name {args.config_name or '<config>'} --prerun {args.prerun}")
    return 0


def _cmd_list(args) -> int:
    from .driver import load_candidates
    from .metrics import assemble_card_metrics, batch_pbo
    from .verdict import assign_badge

    scope = _resolve_scope(args)
    candidates = load_candidates(scope)
    if not candidates:
        print(f"No dumped candidates under {scope.dir / 'landing_examples'} — "
              "run the `run` subcommand first.")
        return 1

    def _f(x, nd=2):
        return f"{x:.{nd}f}" if x is not None else "  –  "

    hdr = (f"{'att':>3} | {'strategy':<32} | {'decision':<9} | {'stage':<12} | "
           f"{'IS_SR':>6} | {'DSR':>5} | {'PBO':>5} | {'OOS':<6} | "
           f"{'badge':<15} | ready")
    print(hdr)
    print("-" * len(hdr))
    for c in candidates:
        m = assemble_card_metrics(c)
        badge = assign_badge(m) if m.card_ready else "–"
        decision = "accept" if c.get("approved") else "reject"
        stage = c.get("reject_stage") or "-"
        pbo = f"{m.pbo:.2f}" if m.pbo is not None else "  –  "
        print(f"{c.get('attempt', '?'):>3} | {m.strategy_name[:32]:<32} | "
              f"{decision:<9} | {stage:<12} | {_f(m.is_sharpe):>6} | "
              f"{_f(m.dsr_prob):>5} | {pbo:>5} | {m.oos_result:<6} | "
              f"{badge:<15} | {'yes' if m.card_ready else 'no'}")

    bp = batch_pbo(candidates)
    if bp["pbo"] is not None:
        print(f"\nBatch-level PBO across {bp['n_configs']} attempts "
              f"({bp['n_splits']} CSCV splits): {bp['pbo']:.2f}")

    if args.auto:
        ready = [(c, assemble_card_metrics(c)) for c in candidates]
        ready = [(c, m) for c, m in ready if m.card_ready]
        if len(ready) >= 2:
            robust = max(ready, key=lambda cm: (cm[1].dsr_prob or 0)
                         - (cm[1].pbo if cm[1].pbo is not None else 1))
            fragile = min(ready, key=lambda cm: (cm[1].dsr_prob or 0)
                          - (cm[1].pbo if cm[1].pbo is not None else 1))
            print(f"\nSuggested max-contrast pair (NOT auto-exported): "
                  f"robust=attempt {robust[0]['attempt']}, "
                  f"fragile=attempt {fragile[0]['attempt']}")
    return 0


def _cmd_export(args) -> int:
    from .driver import load_batch_meta, load_candidates
    from .export import export_example, rebuild_index

    scope = _resolve_scope(args)

    # The OOS recompute must see the same universe/factor scope the run used.
    meta = load_batch_meta(scope) or {}
    n_tickers = (meta.get("env") or {}).get("ARCHITECT_N_TICKERS")
    if n_tickers and not os.getenv("ARCHITECT_N_TICKERS"):
        os.environ["ARCHITECT_N_TICKERS"] = str(n_tickers)
    scope.activate(factor_db=scope.composed_db_path)

    candidates = {c.get("attempt"): c for c in load_candidates(scope)}
    if not candidates:
        print("No dumped candidates — run the `run` subcommand first.")
        return 1

    picks: list[tuple[int, str]] = []
    for spec in args.pick:
        try:
            attempt_s, name = spec.split(":", 1)
            picks.append((int(attempt_s), name))
        except ValueError:
            raise SystemExit(f"--pick expects <attempt>:<example-name>, got {spec!r}")

    out_root = Path(args.out)
    exported = []
    for attempt, name in picks:
        c = candidates.get(attempt)
        if c is None:
            raise SystemExit(f"attempt {attempt} not found "
                             f"(have: {sorted(candidates)})")
        out_dir = export_example(c, name, out_root, scope,
                                 polish_llm=args.polish_llm)
        exported.append(out_dir)

    index = rebuild_index(out_root)
    print(f"\nExported {len(exported)} example(s):")
    for d in exported:
        print(f"  {d}")
    print(f"Index → {index}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="landing_examples",
        description=__doc__.split("\n")[0],
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None,
                        help="Alternate quant.config.yaml (sets QF_CONFIG_FILE).")
    common.add_argument("--config-name", default=None,
                        help="Workspace config scope name "
                             "(default: derived from the active config).")
    common.add_argument("--prerun", required=True,
                        help="Prerun under the config scope whose factors to use.")

    pr = sub.add_parser("run", parents=[common],
                        help="Run N pipeline attempts and dump every candidate.")
    pr.add_argument("--n-strategies", type=int, default=8)
    pr.add_argument("--max-iterations", type=int, default=6,
                    help="Architect refinement iterations per attempt (more "
                         "variants → deflation/PBO have real teeth).")
    pr.add_argument("--oos-ratio", type=float, default=0.2)
    pr.add_argument("--no-seeds", action="store_true",
                    help="Hide seed alphas — strategies use only the prerun's "
                         "researched factors.")
    pr.add_argument("--n-tickers", type=int, default=0,
                    help="Universe cap (sets ARCHITECT_N_TICKERS). 0 = all.")
    pr.add_argument("--target-horizon", type=int, default=None)
    pr.add_argument("--position-construction",
                    choices=["auto", "cross_sectional", "per_underlying"],
                    default="auto")
    pr.set_defaults(func=_cmd_run)

    pl = sub.add_parser("list", parents=[common],
                        help="Table of every dumped candidate's harness metrics.")
    pl.add_argument("--auto", action="store_true",
                    help="Also print a suggested max-contrast pair (never "
                         "auto-exports).")
    pl.set_defaults(func=_cmd_list)

    pe = sub.add_parser("export", parents=[common],
                        help="Export picked candidates as landing-page examples.")
    pe.add_argument("--pick", nargs="+", required=True,
                    metavar="ATTEMPT:NAME",
                    help="e.g. --pick 1:robust-example 4:overfit-example")
    pe.add_argument("--out", required=True,
                    help="Examples root, e.g. ../company-brain/marketing/examples")
    pe.add_argument("--polish-llm", action="store_true",
                    help="One LLM call to smooth the transcript's USER turn "
                         "(flagged in provenance).")
    pe.set_defaults(func=_cmd_export)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
