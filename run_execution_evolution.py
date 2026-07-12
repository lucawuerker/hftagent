#!/usr/bin/env python
"""Run the evolutionary Execution Researcher (E1: jitter-only search).

The execution twin of ``run_factor_evolution.py``: each run lives in a
workspace prerun scope (``data/workspaces/<config>/preruns/<name>/``), the
evolution state under ``<scope>/evolution_exec/``.  The K frozen evaluation
signals are the required input (DESIGN §Cross-signal axis):

* ``--eval-signals manifest:<path>`` — consume an existing FrozenSignalSet
  (the joint outer layer's path), or
* ``--eval-signals prebook:<path>``  — freeze v1 from a prebook JSON of factor
  programs (``research_eval.prebook`` format) before the search starts.

Examples:
    ./venv/bin/python run_execution_evolution.py --name exec-base \\
        --eval-signals prebook:data/books/base/prebook.json \\
        --generations 5 --population 8

    ./venv/bin/python run_execution_evolution.py --name exec-base --resume \\
        --generations 3          # 3 MORE generations from the checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run_execution_evolution")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="exec-base", help="prerun (scope) name")
    p.add_argument("--config-name", default=None,
                   help="workspace config scope (default: derived from settings)")
    p.add_argument("--reset", action="store_true", help="purge the scope first")
    p.add_argument("--resume", action="store_true",
                   help="continue from <scope>/evolution_exec/state.json")
    p.add_argument("--eval-signals", default=None,
                   help="manifest:<path> or prebook:<path> (required unless --resume)")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--children-per-gen", type=int, default=6)
    p.add_argument("--islands", type=int, default=1)
    p.add_argument("--migration-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime", choices=["cross_sectional", "per_underlying"],
                   default=None, help="restrict seeds to one book shape")
    p.add_argument("--jitter-pct", type=float, default=0.15)
    # operator mix (E2): default jitter-only, no LLM
    p.add_argument("--p-llm", type=float, default=0.0,
                   help="probability weight of the LLM-semantic mutation")
    p.add_argument("--p-crossover", type=float, default=0.0)
    p.add_argument("--p-jitter", type=float, default=1.0)
    p.add_argument("--debate", choices=["on", "off"], default="off")
    p.add_argument("--retrieval", choices=["none", "rag"], default="none")
    # ── evaluation knobs ──
    p.add_argument("--is-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--cutoff-date", default=None)
    p.add_argument("--data-dir", default="ticker_data")
    p.add_argument("--n-tickers", type=int, default=15)
    p.add_argument("--horizon", type=int, default=6,
                   help="target horizon for prebook freezing")
    p.add_argument("--cost-rate", type=float, default=5e-4)
    p.add_argument("--lambda-dispersion", type=float, default=0.5)
    p.add_argument("--gate-turnover", type=float, default=None)
    p.add_argument("--gate-degradation", type=float, default=0.5)
    p.add_argument("--min-activity", type=float, default=0.05)
    p.add_argument("--selection-deflation", choices=["off", "on"], default="off")
    return p


def _resolve_signals(spec: str | None, scope, args) -> str:
    """Turn --eval-signals into a FrozenSignalSet manifest path."""
    if not spec:
        raise SystemExit("--eval-signals is required (manifest:<path> or "
                         "prebook:<path>) unless --resume")
    kind, _, path = spec.partition(":")
    if kind == "manifest":
        return path
    if kind == "prebook":
        from quant_fund_agent.mcp import research_client
        from quant_fund_agent.research_eval.prebook import book_entries, load_prebook

        book = book_entries(load_prebook(path))
        if not book:
            raise SystemExit(f"prebook {path} holds no factor programs")
        out = research_client.freeze_signals(
            book, out_dir=str(scope.dir / "evolution_exec"), version=1,
            target_horizon=args.horizon, is_frac=args.is_frac,
            val_frac=args.val_frac, cutoff_date=args.cutoff_date,
            data_dir=args.data_dir, n_tickers=args.n_tickers)
        if not out.get("ok"):
            raise SystemExit(f"signal freeze failed: {out.get('error')}")
        audit = out["manifest"].get("poison_audit", {})
        if audit.get("passed") is False:
            raise SystemExit("frozen-signal poison audit FAILED — refusing to "
                             "search against leaky evaluation signals")
        log.info("Froze %d evaluation signal(s) → %s (audit passed)",
                 out["manifest"]["k"], out["manifest_path"])
        return out["manifest_path"]
    raise SystemExit(f"unknown --eval-signals kind {kind!r} "
                     "(use manifest:<path> or prebook:<path>)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from quant_fund_agent.agents.execution_research.evolution import (
        ExecEvolutionLoop,
        ExecEvolutionRunConfig,
    )
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.workspace import Scope

    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    scope = Scope(config_name, args.name)
    if args.reset:
        log.info("Resetting scope '%s' …", scope.label)
        scope.purge()
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    os.environ["FACTOR_DB_PATH"] = str(scope.factor_db_path)
    os.environ["QF_SCOPE"] = scope.label

    out_dir = scope.dir / "evolution_exec"
    if args.resume:
        run_cfg_path = out_dir / "run_config.json"
        if not run_cfg_path.exists():
            raise SystemExit(f"--resume: no prior run at {run_cfg_path}")
        prior = json.loads(run_cfg_path.read_text())
        manifest = (args.eval_signals.partition(":")[2]
                    if args.eval_signals else prior["signals_manifest"])
    else:
        manifest = _resolve_signals(args.eval_signals, scope, args)

    cfg = ExecEvolutionRunConfig(
        name=args.name,
        out_dir=str(out_dir),
        signals_manifest=manifest,
        generations=args.generations,
        population_size=args.population,
        children_per_generation=args.children_per_gen,
        n_islands=args.islands,
        migration_every=args.migration_every,
        seed=args.seed,
        regime=args.regime,
        jitter_pct=args.jitter_pct,
        p_llm_semantic=args.p_llm,
        p_crossover=args.p_crossover,
        p_jitter=args.p_jitter,
        debate=args.debate,
        retrieval=args.retrieval,
        is_frac=args.is_frac,
        val_frac=args.val_frac,
        cutoff_date=args.cutoff_date,
        data_dir=args.data_dir,
        n_tickers=args.n_tickers,
        cost_rate=args.cost_rate,
        lambda_dispersion=args.lambda_dispersion,
        gate_turnover=args.gate_turnover,
        gate_degradation=args.gate_degradation,
        min_activity=args.min_activity,
        selection_deflation=args.selection_deflation,
    )
    loop = ExecEvolutionLoop(cfg)
    summary = loop.run(resume=args.resume)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Execution evolution '{scope.label}' complete")
    print(f"  archive: {len(summary['archive'])} executor(s); "
          f"SOTA: {summary['sota_executor']}; n_trials={summary['n_trials']}")
    print(f"  state: {out_dir / 'state.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
