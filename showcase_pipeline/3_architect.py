"""Showcase — Stage 3: Architect Agent (no market data required).

The Architect turns the Selector's hypothesis + factors into a concrete strategy
specification: it picks a model from the toolbox (a static-weight blend or a
fitted ML model such as ridge / random-forest / xgboost), its hyper-parameters,
the feature factors, and the position-construction settings.

This script runs the Selector (Stage 2) and then the Architect's first design
step, stopping right before the refinement loop that needs market data:

    design_model                          (this script)
    ──────────────────────────────────────────────────────────────
    fit_and_backtest → evaluate → revise  (SKIPPED — needs ticker_data/)

The designed strategy is persisted as a DRAFT ``StrategyRecord`` into a
*separate showcase strategy database*.  A strategy in this system is realised as
a ``DynamicStrategy`` (static weights) or ``ModelStrategy`` (fitted ML) subclass,
parameterised by that record — this script rebuilds the subclass from the record
to show it concretely.

What your supervisor can inspect afterwards:
  * the new DRAFT entry in ``data/strategies/showcase_strategy_db.json``

Usage::

    ./venv/bin/python showcase_pipeline/3_architect.py
    ./venv/bin/python showcase_pipeline/3_architect.py --target-horizon 6
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path

import showcase_common as sc  # FIRST import: configures env + working dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target-horizon", type=int, default=6,
                   help="Forecast horizon in 10-sec bars (1=10s, 6=1min, 60=10min).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    sc.banner("SHOWCASE · STAGE 3 · ARCHITECT")
    print("Designs a strategy (model + features + position rules) from the")
    print("Selector's hypothesis.  The fit + backtest refinement loop is SKIPPED")
    print("(no market data here), so the strategy is stored as a DRAFT.")

    sc.require_openai_key()
    sc.ensure_showcase_factor_db()

    from quant_fund_agent.agents.architect.graph import design_model
    from quant_fund_agent.agents.architect.state import ArchitectState
    from quant_fund_agent.agents.selector.graph import selector_graph
    from quant_fund_agent.databases import StrategyDatabase
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.pipeline import _slugify, strategy_from_record
    from quant_fund_agent.schemas import StrategyRecord, StrategyStatus

    discover_factors()

    # ── Run the Selector to get a hypothesis + factor set ────────────────
    sc.section("SELECTOR (input to the Architect)")
    selector_result = selector_graph.invoke({})
    hypothesis = selector_result["hypothesis"]
    selected_ids = selector_result["selected_factor_ids"]
    print(f"  Hypothesis : {hypothesis}")
    print(f"  Factors    : {selected_ids}")

    # ── Architect: design the model (one LLM call, no data) ──────────────
    state = ArchitectState(
        hypothesis=hypothesis,
        selected_factor_ids=selected_ids,
        factor_catalog=selector_result["factor_catalog"],
        target_horizon=args.target_horizon,
    )
    design = design_model(state)
    spec = design["strategy_spec"]

    sc.section("DESIGNED STRATEGY SPEC")
    print(f"  strategy_name  : {spec.strategy_name}")
    print(f"  model_type     : {spec.model_type}")
    print(f"  model_params   : {spec.model_params}")
    print(f"  factor_ids     : {spec.factor_ids}")
    print(f"  weights        : {spec.weights}")
    print(f"  target_horizon : {spec.target_horizon}")
    print(f"  holding_period : {spec.holding_period}")
    print(f"  max_positions  : {spec.max_positions}")
    print(f"  equal_weight   : {spec.equal_weight}")
    print(f"  min_conviction : {spec.min_conviction}")
    print(f"  reasoning      : {spec.reasoning}")

    # ── Build a DRAFT StrategyRecord (no metrics — backtest was skipped) ─
    model_type = spec.model_type or "static_weights"
    is_ml = model_type != "static_weights"
    factor_ids = list(spec.factor_ids) if spec.factor_ids else list(spec.weights.keys())
    sid = f"{_slugify(spec.strategy_name)}_{uuid.uuid4().hex[:8]}"

    record = StrategyRecord(
        id=sid,
        name=spec.strategy_name,
        class_name="ModelStrategy" if is_ml else "DynamicStrategy",
        description=hypothesis,
        factor_ids=factor_ids,
        status=StrategyStatus.DRAFT,         # designed, not yet backtested/approved
        backtest_metrics=None,
        oos_backtest_metrics=None,
        model_type=model_type,
        model_params={
            "model_type": model_type,
            "model_hyperparams": dict(spec.model_params),
            "factor_ids": factor_ids,
            "target_horizon": spec.target_horizon,
            "weights": dict(spec.weights),
            "holding_period": spec.holding_period,
            "max_positions": spec.max_positions,
            "equal_weight": spec.equal_weight,
            "min_conviction": spec.min_conviction,
        },
        holding_period=spec.holding_period,
        metadata={
            "showcase": True,
            "architect_reasoning": spec.reasoning,
            "note": "Designed by the Architect; fit + backtest skipped "
                    "(no market data in this repository).",
        },
    )

    db = StrategyDatabase()
    db.load_from_json(sc.STRATEGY_DB_SHOWCASE)
    db.register_strategy(record, portfolio_returns=None)
    db.save_to_json(sc.STRATEGY_DB_SHOWCASE)

    db_file = sc.STRATEGY_DB_SHOWCASE
    sc.section("STRATEGY ADDED")
    print(f"  ✓ {record.id}  —  {record.name}")
    print(f"      subclass            : {record.class_name} "
          f"({'fitted ML model' if is_ml else 'static-weight blend'})")
    print(f"      SAVED TO DB file    : {db_file.name}")
    print(f"               in folder  : {db_file.parent}/")
    print(f"      entry key (id)      : {record.id}  (status={record.status.value})")

    # ── Show the strategy as a concrete subclass instance ────────────────
    if not is_ml:
        strat = strategy_from_record(record)
        print(f"      rebuilt as   : {type(strat).__name__} instance")
        print(f"      factor_ids   : {strat.factor_ids}")
        print(f"      weights      : {record.model_params['weights']}")
    else:
        print("      (ModelStrategy: the estimator would be fitted on the panel "
              "during the backtest — skipped here. The full spec is persisted.)")

    sc.section("PERSISTED RECORD (JSON)")
    print(json.dumps(record.model_dump(mode="json"), indent=2, default=str))

    sc.note_skipped_data_step(
        "the Architect's fit + backtest refinement loop and the Statistician's "
        "out-of-sample acceptance tests (both need ticker_data/)."
    )

    sc.banner("STAGE 3 COMPLETE")
    print(f"  1 DRAFT strategy ('{record.id}') added to DB file: "
          f"{sc.STRATEGY_DB_SHOWCASE.name}")
    print(f"      in folder: {sc.STRATEGY_DB_SHOWCASE.parent}/")
    print("  In a full run with market data this strategy would next be fitted,")
    print("  backtested, and sent to the Statistician for an accept/reject verdict.")


if __name__ == "__main__":
    main()
