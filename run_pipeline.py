"""Run the Selector → Architect → Statistician pipeline end-to-end.

Usage::

    export OPENAI_API_KEY=sk-...
    python run_pipeline.py
    python run_pipeline.py --max-iterations 3 --oos-ratio 0.25
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%H:%M:%S",
)

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY in .env or environment before running.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max refinement iterations for the architect (default 5)")
    parser.add_argument("--oos-ratio", type=float, default=0.2,
                        help="Fraction of data held out for the statistician (default 0.2)")
    parser.add_argument("--skip-statistician", action="store_true",
                        help="Stop after the architect, don't run the statistician.")
    args = parser.parse_args()

    from quant_fund_agent.agents.architect.graph import architect_graph
    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.agents.statistician.graph import statistician_graph

    # ── Step 1: Selector Agent ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: SELECTOR AGENT")
    print("=" * 80)

    selector_result = selector_graph.invoke({})

    print(f"\nHypothesis: {selector_result['hypothesis']}")
    print(f"Reasoning:  {selector_result['reasoning'][:300]}")
    print(f"Selected:   {selector_result['selected_factor_ids']}")

    # ── Step 2: Architect Agent ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: ARCHITECT AGENT (max iterations: %d, OOS ratio: %.2f)"
          % (args.max_iterations, args.oos_ratio))
    print("=" * 80)

    architect_input = {
        "hypothesis": selector_result["hypothesis"],
        "selected_factor_ids": selector_result["selected_factor_ids"],
        "factor_catalog": selector_result["factor_catalog"],
        "max_iterations": args.max_iterations,
        "oos_split_ratio": args.oos_ratio,
    }

    architect_result = architect_graph.invoke(architect_input)

    # ── Print trial history ───────────────────────────────────────────
    print("\n" + "-" * 80)
    print("TRIAL HISTORY  (total trials: %d)" % len(architect_result["trial_history"]))
    print("-" * 80)
    for trial in architect_result["trial_history"]:
        m = trial.metrics
        print(f"\n  Trial {trial.iteration}:")
        print(f"    Strategy:  {trial.spec.strategy_name}")
        print(f"    Weights:   {trial.spec.weights}")
        print(f"    Hold:      {trial.spec.holding_period}  "
              f"MaxPos: {trial.spec.max_positions}  "
              f"EqW: {trial.spec.equal_weight}  "
              f"MinConv: {trial.spec.min_conviction}")
        print(f"    Sharpe:    {m.get('sharpe_ratio')}")
        print(f"    Sortino:   {m.get('sortino_ratio')}")
        print(f"    MaxDD:     {m.get('max_drawdown')}")
        print(f"    AnnRet:    {m.get('annualised_return')}")
        print(f"    HitRate:   {m.get('hit_rate')}")
        print(f"    IC/ICIR:   {m.get('ic_mean')} / {m.get('ic_ir')}")
        print(f"    Reasoning: {trial.spec.reasoning[:150]}")

    # ── Architect's final decision ────────────────────────────────────
    print("\n" + "-" * 80)
    print("ARCHITECT FINAL DECISION: %s" % architect_result["decision"].upper())
    print("-" * 80)
    print(architect_result["decision_reasoning"])

    if args.skip_statistician or architect_result["decision"] != "approve":
        return

    # ── Step 3: Statistician Agent ────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: STATISTICIAN AGENT")
    print("=" * 80)

    statistician_input = {
        "hypothesis": architect_result["hypothesis"],
        "strategy_spec": architect_result["strategy_spec"],
        "trial_history": architect_result["trial_history"],
        "backtest_metrics": architect_result["backtest_metrics"],
        "initial_reasoning": architect_result.get("initial_reasoning", ""),
        "final_reasoning": architect_result["strategy_spec"].reasoning,
        "oos_split_ratio": args.oos_ratio,
    }

    stat_result = statistician_graph.invoke(statistician_input)

    # ── Print test results ────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("STATISTICAL TESTS")
    print("-" * 80)
    for r in stat_result["test_results"]:
        print(f"\n  [{r.verdict.value.upper()}] {r.test_name}")
        print(f"    value={r.value}  threshold={r.threshold}  p={r.p_value}")
        print(f"    {r.summary}")
        if r.details:
            for k, v in r.details.items():
                if isinstance(v, dict):
                    continue
                print(f"      {k}: {v}")

    # ── Risk assessment + final decision ──────────────────────────────
    print("\n" + "-" * 80)
    print("RISK ASSESSMENT")
    print("-" * 80)
    print(stat_result.get("risk_assessment", "(none)"))

    print("\n" + "-" * 80)
    print("STATISTICIAN FINAL DECISION: %s"
          % stat_result["final_decision"].upper())
    print("-" * 80)
    print(stat_result["final_reasoning_statistician"])


if __name__ == "__main__":
    main()
