"""Evolutionary factor research: closed-loop Pareto search over factor programs.

The ``--engine evolution`` counterpart to ``run_factor_research.py`` (which stays
the ``oneshot`` baseline arm).  The LLM mutates Pareto-selected parents; the
deterministic ``research_eval`` harness scores every candidate out-of-sample
(IS fit / VAL fitness / TEST untouched, CPCV robustness, N_trials-aware
deflation); the controller keeps a gate-passing Pareto archive which *is* the
accepted book.  See ``docs/research-evolution/DESIGN.md``.

Like every research run, an evolution run is a **prerun**: it persists its final
archive into ``data/workspaces/<config>/preruns/<name>/`` (plus the full
evolution state + lineage under ``<scope>/evolution/``), so the comparison and
rolling harnesses can score it against oneshot preruns directly.

Examples
--------
::

    # A small evolution run with the default research model.
    python run_factor_evolution.py --name evo1 --generations 5 --population 10

    # Larger run, seeded from papers, two islands, stronger model.
    python run_factor_evolution.py --name evo_claude --model claude-sonnet-5 \
      --llm-provider anthropic --generations 12 --population 12 \
      --children-per-gen 10 --islands 2 --seed-papers 6

    # Smoke run, in-process, tiny.
    QF_USE_MCP=0 python run_factor_evolution.py --name evosmoke --reset \
      --generations 2 --population 4 --children-per-gen 3 --seed-ideas 4 \
      --n-tickers 8
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
log = logging.getLogger("run_factor_evolution")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # ── scope / model (mirrors run_factor_research.py) ──
    p.add_argument("--name", default=None,
                   help="Prerun id within the config scope (default 'base').")
    p.add_argument("--config-name", default=None,
                   help="Config scope under data/workspaces/ (default: derived "
                        "from the active config).")
    p.add_argument("--model", default=None,
                   help="Research LLM model id (sets FACTOR_RESEARCH_LLM_MODEL).")
    p.add_argument("--llm-provider", default=None)
    p.add_argument("--reset", action="store_true",
                   help="Purge this prerun's factors + evolution state first.")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--n-tickers", type=int, default=15)

    # ── search shape ──
    p.add_argument("--evolution-unit", choices=["single", "set"], default="single",
                   help="Evolve one factor program (scored by what it adds to the "
                        "archive) or a whole set/'alpha program' (scored jointly).")
    p.add_argument("--set-size", type=int, default=3,
                   help="Initial members per SET genome (set mode only).")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--population", type=int, default=10, help="Per-island size.")
    p.add_argument("--children-per-gen", type=int, default=8)
    p.add_argument("--islands", type=int, default=1)
    p.add_argument("--migration-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the controller.")

    # ── operator mix ──
    p.add_argument("--p-llm", type=float, default=0.6, help="P(llm_semantic).")
    p.add_argument("--p-crossover", type=float, default=0.25)
    p.add_argument("--p-jitter", type=float, default=0.15)

    # ── seeding ──
    p.add_argument("--seed-ideas", type=int, default=8,
                   help="Generation-0 brainstorm budget.")
    p.add_argument("--seed-papers", type=int, default=0,
                   help="Papers pulled for the seed brainstorm (0 = knowledge-only).")

    # ── retrieval (P2) ──
    p.add_argument("--retrieval", choices=["none", "rag", "graphrag"], default="none",
                   help="Ground the brainstorm in retrieved papers (rag) or the "
                        "hybrid knowledge graph (graphrag, P4).")
    p.add_argument("--retrieval-cardinality", choices=["1toN", "Nto1", "NtoM"],
                   default="1toN",
                   help="1toN: one paper per call; Nto1: cross-paper synthesis; "
                        "NtoM: N papers, M ideas.")
    p.add_argument("--rag-k", type=int, default=4,
                   help="Papers retrieved per grounded brainstorm.")

    # ── agent split + debate (P3) ──
    p.add_argument("--debate", choices=["on", "off"], default="off",
                   help="on: Hypothesis → skeptic/moderator Debate → Codegen "
                        "(weak ideas die before codegen); off: single-call "
                        "mutation (the ablation arm).")
    p.add_argument("--hypothesis-model", default=None,
                   help="Model for the Hypothesis / mutation role "
                        "(default: --model).")
    p.add_argument("--debate-model", default=None,
                   help="Model for the skeptic/moderator (default: --model).")
    p.add_argument("--codegen-model", default=None,
                   help="Model for codegen (default: --model).")

    # ── evaluation / overfit control ──
    p.add_argument("--horizon", type=int, default=6,
                   help="Combined-model forecast horizon (bars).")
    p.add_argument("--is-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--cpcv-folds", type=int, default=6, dest="cpcv_groups",
                   help="CPCV group count (C(N,k) folds).")
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--embargo", type=int, default=0)
    p.add_argument("--cutoff-date", default=None,
                   help="ISO date: evaluate only bars strictly before this "
                        "(walk-forward wrapping).")
    p.add_argument("--walk-forward", default=None,
                   help="Comma list of ascending ISO dates d0,d1,…: re-run the "
                        "WHOLE loop per fold (evolve < d_i, touch-once score on "
                        "[d_i, d_{i+1})).  Validation only — nothing is "
                        "persisted to the prerun's factor DB.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.model:
        os.environ["FACTOR_RESEARCH_LLM_MODEL"] = args.model
    if args.llm_provider:
        os.environ["FACTOR_RESEARCH_LLM_PROVIDER"] = args.llm_provider
    # Per-role model overrides (P3): tokens go where they matter.
    for role, value in (("HYPOTHESIS", args.hypothesis_model),
                        ("DEBATE", args.debate_model),
                        ("CODEGEN", args.codegen_model)):
        if value:
            os.environ[f"{role}_LLM_MODEL"] = value
    os.environ["DATA_DIR"] = args.data_dir

    from quant_fund_agent.agents.factor_research.evolution.loop import (
        EvolutionLoop,
        EvolutionRunConfig,
        persist_archive,
    )
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.llm import resolve_research_model, resolve_research_provider
    from quant_fund_agent.workspace import Scope

    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    prerun = args.name or "base"
    scope = Scope(config_name, prerun)
    if args.reset:
        log.info("Resetting scope '%s' …", scope.label)
        scope.purge()
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    os.environ["FACTOR_DB_PATH"] = str(scope.factor_db_path)
    os.environ["PAPER_READ_LOG"] = str(scope.read_log_path)
    os.environ["QF_SCOPE"] = scope.label

    model = resolve_research_model()
    provider = resolve_research_provider(model)
    log.info("Evolution run into scope '%s' (model=%s, provider=%s)",
             scope.label, model, provider)

    cfg = EvolutionRunConfig(
        generations=args.generations,
        population_size=args.population,
        children_per_generation=args.children_per_gen,
        n_islands=args.islands,
        migration_every=args.migration_every,
        seed=args.seed,
        unit=args.evolution_unit,
        set_size=args.set_size,
        p_llm_semantic=args.p_llm,
        p_crossover=args.p_crossover,
        p_jitter=args.p_jitter,
        n_seed_ideas=args.seed_ideas,
        seed_papers=args.seed_papers,
        retrieval=args.retrieval,
        retrieval_cardinality=args.retrieval_cardinality,
        rag_k=args.rag_k,
        debate=args.debate,
        target_horizon=args.horizon,
        is_frac=args.is_frac,
        val_frac=args.val_frac,
        cpcv_groups=args.cpcv_groups,
        cpcv_k=args.cpcv_k,
        embargo=args.embargo,
        cutoff_date=args.cutoff_date,
        data_dir=args.data_dir,
        n_tickers=args.n_tickers,
        out_dir=str(scope.dir / "evolution"),
    )

    if args.walk_forward:
        from quant_fund_agent.agents.factor_research.evolution.walkforward import (
            run_walk_forward,
        )

        boundaries = [d.strip() for d in args.walk_forward.split(",") if d.strip()]
        report = run_walk_forward(cfg, boundaries,
                                  out_dir=scope.dir / "evolution" / "walkforward")
        print("\n" + "=" * 80)
        print(f"Walk-forward validation for scope '{scope.label}' complete")
        print(json.dumps({k: v for k, v in report.items() if k != "folds"},
                         indent=2, default=str))
        print(f"Full per-fold report: "
              f"{scope.dir / 'evolution' / 'walkforward' / 'walkforward.json'}")
        print("=" * 80)
        return

    loop = EvolutionLoop(cfg)
    summary = loop.run()
    session_id = f"evolution:{config_name}:{prerun}"
    persisted = persist_archive(
        loop.controller, session_id=session_id,
        target_horizon=args.horizon, cutoff_date=args.cutoff_date,
        data_dir=args.data_dir, n_tickers=args.n_tickers)
    summary["persisted_factor_ids"] = persisted["kept_factor_ids"]

    scope.write_manifest(
        llm_model=model, llm_provider=provider, engine="evolution",
        generations=summary["generations"], n_trials=summary["n_trials"],
        n_factors=len(persisted["kept_factor_ids"]),
    )

    print("\n" + "=" * 80)
    print(f"Evolution run '{scope.label}' complete")
    print(json.dumps(summary, indent=2, default=str))
    print("=" * 80)


if __name__ == "__main__":
    main()
