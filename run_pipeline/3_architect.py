"""Stage 3: Architect Agent — design + fit + backtest on the local market data.

Runs the Selector to get a hypothesis + factor set, then the **full** Architect
refinement loop on the local panel:

    design_model → fit_and_backtest → evaluate → revise   (× up to --max-iterations)

i.e. the model is actually fitted and backtested in-sample on real data — the step
the showcase version skips.  This is a demonstration of the Architect alone: it
prints the per-iteration trial history and the final accept/revise decision but
does **not** persist a strategy (that happens once the Statistician accepts — see
``4_statistician.py`` and ``run_pipeline.py``).

Stages 3 and 4 each re-run the Selector + Architect (the per-step scripts are
intentionally self-contained, like ``showcase_pipeline/3_architect.py``).  To do
one chained pass without the repeat, use ``run_pipeline.py``.

Usage::

    ./venv/bin/python run_pipeline/3_architect.py --n-tickers 8
    ./venv/bin/python run_pipeline/3_architect.py --max-iterations 3 --target-horizon 6
"""

from __future__ import annotations

import argparse
import logging

import pipeline_common as pc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Architect refinement iterations.")
    p.add_argument("--oos-ratio", type=float, default=0.2,
                   help="Fraction of data held out (in-sample backtest split).")
    p.add_argument("--target-horizon", type=int, default=6,
                   help="Forecast horizon in 10-sec bars (1=10s, 6=1min, 60=10min).")
    pc.add_common_args(p)  # --mcp, --n-tickers, --data-dir
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    pc.banner("STAGE 3 · ARCHITECT")
    print("Selector hypothesis → design a strategy → fit + backtest → revise.")

    pc.require_openai_key()
    pc.require_market_data(args.data_dir)
    pc.configure_runtime(n_tickers=args.n_tickers, data_dir=args.data_dir,
                         use_mcp=args.mcp)

    from quant_fund_agent import pipeline

    res = pipeline.run_strategy_pipeline(
        max_iterations=args.max_iterations,
        oos_ratio=args.oos_ratio,
        target_horizon=args.target_horizon,
        run_statistician=False,
    )

    arch = res.architect_result or {}

    pc.section("SELECTOR (input to the Architect)")
    print(f"  Hypothesis : {res.hypothesis}")
    print(f"  Factors    : {res.selector_result.get('selected_factor_ids')}")

    trials = arch.get("trial_history", [])
    pc.section(f"TRIAL HISTORY ({len(trials)} iteration(s))")
    for t in trials:
        m = t.metrics or {}
        diag = m.get("diagnostics") or {}
        n_feat = len(t.spec.factor_ids or t.spec.weights)
        print(f"\n  Trial {t.iteration}: {t.spec.strategy_name}")
        print(f"    model      : {t.spec.model_type}  ({n_feat} features)")
        print(f"    params     : {t.spec.model_params}")
        print(f"    IS Sharpe  : {pc.fmt(m.get('sharpe_ratio'))}   "
              f"AnnRet: {pc.fmt(m.get('annualised_return'))}   "
              f"MaxDD: {pc.fmt(m.get('max_drawdown'))}")
        print(f"    IC / ICIR  : {pc.fmt(m.get('ic_mean'), 4)} / {pc.fmt(m.get('ic_ir'), 4)}"
              + (f"   valid_ic: {pc.fmt(diag.get('valid_ic'), 4)}" if diag else ""))
        if m.get("error"):
            print(f"    error      : {m['error']}")

    pc.section(f"ARCHITECT DECISION: {str(arch.get('decision', '?')).upper()}")
    print(arch.get("decision_reasoning", "(none)"))

    spec = res.strategy_spec
    if spec is not None:
        pc.section("FINAL STRATEGY SPEC")
        print(f"  strategy_name  : {spec.strategy_name}")
        print(f"  model_type     : {spec.model_type}")
        print(f"  model_params   : {spec.model_params}")
        print(f"  factor_ids     : {spec.factor_ids}")
        print(f"  weights        : {spec.weights}")
        print(f"  target_horizon : {spec.target_horizon}")
        print(f"  holding_period : {spec.holding_period}")
        print(f"  max_positions  : {spec.max_positions}")

    pc.banner("STAGE 3 COMPLETE")
    if res.approved:
        print("  Architect APPROVED the design (in-sample).  Next, the")
        print("  Statistician runs out-of-sample acceptance tests:")
        print("  ./venv/bin/python run_pipeline/4_statistician.py")
    else:
        print("  Architect did not approve a design this run.")


if __name__ == "__main__":
    main()
