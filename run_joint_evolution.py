#!/usr/bin/env python
"""Run the joint factor×execution evolution (the block-coordinate outer layer).

One joint run = one workspace prerun; state under
``data/workspaces/<config>/preruns/<name>/joint/``.  Design:
``docs/joint-evolution/DESIGN.md``.

Examples:
    ./venv/bin/python run_joint_evolution.py --name joint-base \\
        --total-blocks 4 --gens-per-block 2 --scheduler round_robin

    ./venv/bin/python run_joint_evolution.py --name joint-base \\
        --total-blocks 6            # resumes at the checkpointed block
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run_joint_evolution")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="joint-base")
    p.add_argument("--config-name", default=None)
    p.add_argument("--reset", action="store_true")
    # ── outer layer ──
    p.add_argument("--total-blocks", type=int, default=4)
    p.add_argument("--gens-per-block", type=int, default=2)
    p.add_argument("--scheduler",
                   choices=["sequential", "round_robin", "random", "bandit"],
                   default="round_robin")
    p.add_argument("--n-factor-blocks", type=int, default=None,
                   help="sequential split point (default: half)")
    p.add_argument("--coupling", choices=["off", "on"], default="off",
                   help="J3: executor-aware factor cost gate")
    p.add_argument("--bandit-context", choices=["on", "off"], default="on",
                   help="J2: off → non-contextual Gaussian TS")
    p.add_argument("--seed", type=int, default=0)
    # ── shared evaluation frame ──
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--is-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--cutoff-date", default=None)
    p.add_argument("--data-dir", default="ticker_data")
    p.add_argument("--n-tickers", type=int, default=15)
    p.add_argument("--cost-rate", type=float, default=5e-4)
    # ── factor arm (subset of run_factor_evolution's surface) ──
    p.add_argument("--factor-population", type=int, default=12)
    p.add_argument("--factor-children", type=int, default=6)
    p.add_argument("--p-llm", type=float, default=0.5)
    p.add_argument("--p-crossover", type=float, default=0.2)
    p.add_argument("--p-jitter", type=float, default=0.3)
    p.add_argument("--seed-ideas", type=int, default=6)
    p.add_argument("--retrieval", choices=["none", "rag", "graphrag"],
                   default="none")
    p.add_argument("--debate", choices=["on", "off"], default="off")
    # ── exec arm ──
    p.add_argument("--exec-population", type=int, default=8)
    p.add_argument("--exec-children", type=int, default=6)
    p.add_argument("--exec-p-llm", type=float, default=0.5)
    p.add_argument("--exec-p-jitter", type=float, default=0.5)
    p.add_argument("--gate-turnover", type=float, default=None)
    p.add_argument("--walk-forward", default=None,
                   help="J4: comma list of fold boundary dates d0,d1,… — re-runs "
                        "the WHOLE joint loop per fold (cutoff=d_i) and "
                        "touch-once scores each archive on [d_i, d_{i+1})")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from quant_fund_agent.agents.execution_research.evolution import (
        ExecEvolutionRunConfig,
    )
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionRunConfig,
    )
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.joint_evolution.loop import (
        JointEvolutionLoop,
        JointRunConfig,
    )
    from quant_fund_agent.workspace import Scope

    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    scope = Scope(config_name, args.name)
    if args.reset:
        scope.purge()
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    os.environ["FACTOR_DB_PATH"] = str(scope.factor_db_path)
    os.environ["PAPER_READ_LOG"] = str(scope.read_log_path)
    os.environ["QF_SCOPE"] = scope.label

    fields = sorted(usable_fields(settings))
    from quant_fund_agent.agents.factor_research.prompts import build_data_context

    data_context = build_data_context(fields)

    cfg = JointRunConfig(
        out_dir=str(scope.joint_dir),
        total_blocks=args.total_blocks,
        gens_per_block=args.gens_per_block,
        scheduler=args.scheduler,
        n_factor_blocks=args.n_factor_blocks,
        seed=args.seed,
        coupling=(args.coupling == "on"),
        bandit_context=args.bandit_context,
        target_horizon=args.horizon,
        is_frac=args.is_frac, val_frac=args.val_frac,
        cutoff_date=args.cutoff_date,
        data_dir=args.data_dir, n_tickers=args.n_tickers, fields=fields,
        cost_rate=args.cost_rate,
    )
    factor_cfg = EvolutionRunConfig(
        generations=args.gens_per_block,   # overridden per block anyway
        population_size=args.factor_population,
        children_per_generation=args.factor_children,
        seed=args.seed,
        p_llm_semantic=args.p_llm, p_crossover=args.p_crossover,
        p_jitter=args.p_jitter,
        n_seed_ideas=args.seed_ideas, retrieval=args.retrieval,
        debate=args.debate,
        target_horizon=args.horizon, is_frac=args.is_frac,
        val_frac=args.val_frac, cutoff_date=args.cutoff_date,
        data_dir=args.data_dir, n_tickers=args.n_tickers,
        out_dir=str(scope.joint_dir / "factor"),
    )
    exec_cfg = ExecEvolutionRunConfig(
        name=args.name, out_dir=str(scope.joint_dir / "exec"),
        signals_manifest="(joint)",       # provided per block by the outer loop
        population_size=args.exec_population,
        children_per_generation=args.exec_children,
        seed=args.seed,
        p_llm_semantic=args.exec_p_llm, p_jitter=args.exec_p_jitter,
        is_frac=args.is_frac, val_frac=args.val_frac,
        cutoff_date=args.cutoff_date, data_dir=args.data_dir,
        n_tickers=args.n_tickers, cost_rate=args.cost_rate,
        gate_turnover=args.gate_turnover,
    )

    if args.walk_forward:
        from quant_fund_agent.joint_evolution.walkforward import (
            run_joint_walk_forward,
        )

        boundaries = [d.strip() for d in args.walk_forward.split(",") if d.strip()]
        result = run_joint_walk_forward(
            cfg, factor_cfg, exec_cfg, boundaries,
            out_dir=scope.joint_dir / "walkforward",
            data_context=data_context, fields=fields)
        print(f"Joint walk-forward '{scope.label}' complete: "
              f"{result['n_scored']}/{result['n_folds']} folds scored, "
              f"mean OOS net Sharpe={result['mean_oos_net_sharpe']}")
        print(f"  results: {scope.joint_dir / 'walkforward' / 'walkforward.json'}")
        return 0

    loop = JointEvolutionLoop(cfg, factor_cfg, exec_cfg,
                              data_context=data_context, fields=fields)
    summary = loop.run()
    (scope.joint_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"Joint evolution '{scope.label}' complete: "
          f"{summary['blocks']} blocks ({summary['scheduler']}), "
          f"J={summary['J']}, book={summary['book_size']}, "
          f"SOTA executor={summary['sota_executor']}")
    print(f"  state: {scope.joint_dir / 'joint_state.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
