"""GP factor-mining benchmark: a non-LLM baseline for the evolutionary researcher.

Genetic programming over the project's **base grammar** (``factors.ops.BASE_OPS``
+ arithmetic).  Deterministic AutoAlpha-style search — random-tree seeding,
subtree crossover, subtree/point/hoist mutation, hierarchical depth growth —
scored by the **same** ``research_eval`` harness, NSGA-II controller and prerun
persistence path the LLM arm uses, so the two are directly comparable — the same
four maximised objectives (``marginal_value``, ``independence``, ``parsimony``,
``structural_novelty``) behind the same coverage/cost gates.

Diversity structure is the one deliberate difference: the LLM arm organises its
population into knowledge-graph **mechanism groups** each holding several demes,
which the GP arm cannot do (it has no knowledge graph, so there is nothing to
group *by*).  The GP therefore runs a single mechanism group containing
``--islands`` classic flat demes with the usual ring migration.

Like every research run it is a **prerun**: it persists its final Pareto archive
into ``data/workspaces/<config>/preruns/<name>/`` (plus the evolution state +
lineage under ``<scope>/gp/``), so the comparison harness can score it against
oneshot / evolution preruns directly:

    python run_model_comparison.py --preruns <gp-name>,<llm-name> --fast

Examples
--------
::

    # A quick, in-process GP run (no LLM, no server).
    python run_gp_factor_mining.py --name gp1 --config-name yfinance_equity_demo \
        --generations 3 --depth-schedule 3,5 --seed-pop 30 \
        --children-per-generation 12 --n-tickers 10

    # Larger run driven by a data config file.
    python run_gp_factor_mining.py --name gp-sp100 --config quant.config.sp100.yaml \
        --generations 6 --depth-schedule 3,5,7 --seed-pop 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_gp_factor_mining")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # ── scope / data config ──
    p.add_argument("--name", default=None,
                   help="Prerun id within the config scope (default 'gp').")
    p.add_argument("--config-name", default=None,
                   help="Config scope under data/workspaces/ (default: derived "
                        "from the active config).")
    p.add_argument("--config", default=None,
                   help="Path to a quant.config.<x>.yaml (sets QF_CONFIG_FILE "
                        "before the panel loads, like run_backtest.py).")
    p.add_argument("--reset", action="store_true",
                   help="Purge this prerun's factors + GP state first.")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--n-tickers", type=int, default=15)

    # ── search shape ──
    p.add_argument("--generations", type=int, default=5,
                   help="Generations PER depth stage.")
    p.add_argument("--population", type=int, default=20, help="Per-island size.")
    p.add_argument("--seed-pop", type=int, default=40,
                   help="Generation-0 random-tree budget.")
    p.add_argument("--children-per-generation", type=int, default=16)
    p.add_argument("--islands", type=int, default=1,
                   help="Independently evolving demes.  The GP arm has no "
                        "knowledge graph, so it runs a single mechanism group "
                        "and these are classic flat islands within it.")
    p.add_argument("--migration-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--depth-schedule", default="3,5",
                   help="Comma list of max tree depths per stage (AutoAlpha "
                        "hierarchical growth), e.g. '3,5,7'.")

    # ── GP operator mix ──
    p.add_argument("--p-crossover", type=float, default=0.5)
    p.add_argument("--p-subtree", type=float, default=0.25)
    p.add_argument("--p-point", type=float, default=0.15)
    p.add_argument("--p-hoist", type=float, default=0.10)

    # ── evaluation / overfit control (identical knobs to the LLM arm) ──
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--is-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--stability-blocks", type=int, default=4,
                   help="Number of contiguous VAL blocks used to measure the "
                        "stability of the fixed marginal predictions (default 4; "
                        "the marginal models are not refitted).")
    p.add_argument("--cutoff-date", default=None)

    # ── fitness axes (the same four the LLM arm maximises) ──
    p.add_argument("--independence-metric",
                   choices=["residual_ic", "delta_participation"],
                   default="residual_ic")
    p.add_argument("--marginal-model", default="gradient_boosting")
    p.add_argument("--gate-turnover", type=float, default=None)
    p.add_argument("--cost-rate", type=float, default=5e-4)
    p.add_argument("--perturbation-weight", type=float, default=0.0)
    p.add_argument("--perturbation-sigma", type=float, default=0.5)

    # ── curation / publish deflation ──
    p.add_argument("--curation", choices=["archive", "greedy", "elastic_net"],
                   default="archive")
    p.add_argument("--n-keep", type=int, default=None)
    p.add_argument("--selection-deflation", choices=["off", "on"], default="off")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # A data config file must be set before the panel (and settings) load.
    if args.config:
        os.environ["QF_CONFIG_FILE"] = args.config
    os.environ["DATA_DIR"] = args.data_dir
    # Score in-process by default (no MCP server needed); byte-identical numbers.
    os.environ.setdefault("QF_USE_MCP", "0")

    from quant_fund_agent.agents.factor_research.evolution.loop import persist_archive
    from quant_fund_agent.agents.factor_research.gp.loop import GPLoop, GPRunConfig
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.workspace import Scope

    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    prerun = args.name or "gp"
    scope = Scope(config_name, prerun)
    if args.reset:
        log.info("Resetting scope '%s' …", scope.label)
        scope.purge()
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    # Stamp this scope's paths into the environment (researcher-only FACTOR_DB_PATH,
    # returns/model dirs, QF_SCOPE) so persist writes into the prerun, not main.
    scope.activate()

    allowed = sorted(usable_fields())
    log.info("GP run into scope '%s' — %d in-scope fields, grammar confined to "
             "BASE_OPS", scope.label, len(allowed))

    depth_schedule = tuple(int(d) for d in str(args.depth_schedule).split(",")
                           if d.strip())

    cfg = GPRunConfig(
        generations=args.generations,
        population_size=args.population,
        children_per_generation=args.children_per_generation,
        n_islands=args.islands,
        migration_every=args.migration_every,
        seed=args.seed,
        seed_pop=args.seed_pop,
        depth_schedule=depth_schedule,
        p_crossover=args.p_crossover,
        p_subtree=args.p_subtree,
        p_point=args.p_point,
        p_hoist=args.p_hoist,
        target_horizon=args.horizon,
        is_frac=args.is_frac,
        val_frac=args.val_frac,
        stability_blocks=args.stability_blocks,
        cutoff_date=args.cutoff_date,
        independence_metric=args.independence_metric,
        marginal_model=args.marginal_model,
        gate_turnover=args.gate_turnover,
        cost_rate=args.cost_rate,
        perturbation_weight=args.perturbation_weight,
        perturbation_sigma=args.perturbation_sigma,
        curation=args.curation,
        n_keep=args.n_keep,
        selection_deflation=args.selection_deflation,
        data_dir=args.data_dir,
        n_tickers=args.n_tickers,
        out_dir=str(scope.dir / "gp"),
    )

    loop = GPLoop(cfg, allowed)
    summary = loop.run()

    session_id = f"gp:{config_name}:{prerun}"
    persisted = persist_archive(
        loop.controller, session_id=session_id,
        target_horizon=args.horizon, cutoff_date=args.cutoff_date,
        data_dir=args.data_dir, n_tickers=args.n_tickers,
        curation=args.curation, n_keep=args.n_keep,
        is_frac=args.is_frac, val_frac=args.val_frac, fields=loop.fields,
        marginal_model=args.marginal_model,
        selection_deflation=args.selection_deflation,
        engine="gp", model_label="gp")
    summary["persisted_factor_ids"] = persisted["kept_factor_ids"]

    scope.write_manifest(
        llm_model="gp", llm_provider="none", engine="gp",
        generations=summary["generations"], n_trials=summary["n_trials"],
        n_factors=len(persisted["kept_factor_ids"]),
    )

    print("\n" + "=" * 80)
    print(f"GP factor-mining run '{scope.label}' complete")
    print(json.dumps(summary, indent=2, default=str))
    print("=" * 80)


if __name__ == "__main__":
    main()
