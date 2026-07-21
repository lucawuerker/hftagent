"""Evolutionary factor research: closed-loop Pareto search over factor programs.

The ``--engine evolution`` counterpart to ``run_factor_research.py`` (which stays
the ``oneshot`` baseline arm).  The LLM mutates Pareto-selected parents; the
deterministic ``research_eval`` harness scores every candidate out-of-sample
(IS fit / VAL fitness / TEST untouched, blocked validation stability, N_trials-aware
deflation); the controller keeps a gate-passing Pareto archive which *is* the
accepted book.  See ``docs/research-evolution/DESIGN.md``.

Like every research run, an evolution run is a **prerun**: it persists its final
archive into ``data/workspaces/<config>/preruns/<name>/`` (plus the full
evolution state + lineage under ``<scope>/evolution/``), so the comparison and
rolling harnesses can score it against oneshot preruns directly.

Examples
--------
::

    # A grouped evolution run with the default research model.
    python run_factor_evolution.py --name evo1

    # Override the serious defaults for a smaller grouped run.
    python run_factor_evolution.py --name evo_claude --model claude-sonnet-5 \
      --llm-provider anthropic --generations 8 --population 12 \
      --mechanism-groups 4 --demes-per-group 2 --children-per-deme 3

    # Smoke run, in-process, tiny.
    QF_USE_MCP=0 python run_factor_evolution.py --name evosmoke --reset \
      --generations 2 --population 4 --mechanism-groups 1 \
      --demes-per-group 1 --children-per-deme 2 --seed-ideas-per-group 4 \
      --retrieval none \
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
    p.add_argument("--reference-book", default=None,
                   help="Prebook file of reference factors (e.g. the ~86 base "
                        "factors) for the novelty DIAG: the candidate's max-|corr| "
                        "and code distance vs this set (WS4). Diagnostic only, no gate.")
    p.add_argument("--memory", action="store_true",
                   help="Enable the per-config cross-run experience memory (WS5): "
                        "accumulate survivors + per-mechanism attempt/survival tallies "
                        "across runs; steer next-run seeding away from exhausted "
                        "mechanisms and feed the teacher channel.")
    p.add_argument("--fixed-book", default=None,
                   help="JSON prebook to condition SINGLE-mode fitness on without "
                        "inserting those factors into the archive, e.g. a Lasso "
                        "prebook built by scripts/build_lasso_prebook.py.")

    # ── search shape ──
    p.add_argument("--evolution-unit", choices=["single", "set"], default="single",
                   help="Evolve one factor program (scored by what it adds to the "
                        "archive) or a whole set/'alpha program' (scored jointly).")
    p.add_argument("--set-size", type=int, default=3,
                   help="Initial members per SET genome (set mode only).")
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--population", type=int, default=16, help="Per-deme size.")
    p.add_argument("--mechanism-groups", type=int, default=5,
                   help="Reserved knowledge-graph mechanism communities.")
    p.add_argument("--demes-per-group", type=int, default=3,
                   help="Classic independently evolving islands within each group.")
    p.add_argument("--children-per-deme", type=int, default=4,
                   help="Children proposed by every deme in every generation.")
    p.add_argument("--children-per-gen", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--islands", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--migration-every", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the controller.")
    # ── operator mix ──
    p.add_argument("--p-llm", type=float, default=0.55, help="P(llm_semantic).")
    p.add_argument("--p-crossover", type=float, default=0.25)
    p.add_argument("--p-cross-group", type=float, default=0.10,
                   help="P(crossover using parents from different mechanism groups).")
    p.add_argument("--p-jitter", type=float, default=0.10)

    # ── seeding ──
    p.add_argument("--seed-ideas-per-group", type=int, default=6,
                   help="Generation-0 brainstorm budget for every mechanism group.")
    p.add_argument("--seed-papers", type=int, default=0,
                   help="Papers pulled for the seed brainstorm (0 = knowledge-only).")

    # ── retrieval (P2) ──
    p.add_argument("--retrieval", choices=["none", "rag", "graphrag"], default="graphrag",
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
    p.add_argument("--stability-blocks", type=int, default=4,
                   help="Number of contiguous VAL blocks used to measure the "
                        "stability of the fixed marginal predictions (default 4; "
                        "the marginal models are not refitted).")
    p.add_argument("--prediction-horizon-mode", choices=["fixed", "free"],
                   default="fixed",
                   help="fixed (default): force every generated factor to declare "
                        "--horizon as prediction_horizon. free: let the researcher "
                        "declare per-factor horizons, while fitness still scores at "
                        "--horizon.")
    p.add_argument("--cutoff-date", default=None,
                   help="ISO date: evaluate only bars strictly before this "
                        "(walk-forward wrapping).")

    # ── progressive data reveal (default OFF) ──
    p.add_argument("--progressive-reveal", action="store_true",
                   help="Reveal the dev window block-by-block across generations "
                        "(expanding IS + sliding VAL), so each generation is partly "
                        "scored on data no earlier selection saw — prevention of the "
                        "generational overfitting ratchet.  Ignores --is-frac/"
                        "--val-frac; the final --test-frac tail is never revealed.")
    p.add_argument("--test-frac", type=float, default=0.2,
                   help="Progressive mode: the final never-revealed TEST tail "
                        "fraction (default 0.2).")
    p.add_argument("--seed-frac", type=float, default=0.45,
                   help="Progressive mode: fraction of the dev window visible at "
                        "generation 0 (default 0.45).")
    p.add_argument("--reveal-every", type=int, default=1,
                   help="Progressive mode: generations between block reveals "
                        "(default 1 = reveal every generation).")
    p.add_argument("--val-blocks", type=int, default=2,
                   help="Progressive mode: sliding VAL spans the last this-many "
                        "revealed blocks (default 2).")
    p.add_argument("--walk-forward", default=None,
                   help="Comma list of ascending ISO dates d0,d1,…: re-run the "
                        "WHOLE loop per fold (evolve < d_i, touch-once score on "
                        "[d_i, d_{i+1})).  Validation only — nothing is "
                        "persisted to the prerun's factor DB.")

    # ── fitness axes ──
    p.add_argument("--independence-metric",
                   choices=["residual_ic", "delta_participation"],
                   default="residual_ic",
                   help="Independence axis basis: residual (orthogonalised) IC "
                        "(default; novel predictive content) or the legacy "
                        "Δ-participation-ratio − max-|corr| penalty.")
    p.add_argument("--marginal-model", default="gradient_boosting",
                   help="Estimator that combines the book for the marginal-value "
                        "(LOCO) axis. Default 'gradient_boosting' is NONLINEAR so "
                        "conditioning/interaction factors (e.g. a volatility state "
                        "variable, valuable only via vol×momentum) score a positive "
                        "marginal value; 'ridge' = additive-only (an ablation). Any "
                        "modeling.catalog id works (random_forest, xgboost, …).")
    # ── economic realism (P5, WS3): folded in, no new axis; all default OFF ──
    p.add_argument("--gate-turnover", type=float, default=None,
                   help="Enable the cost_ok gate: reject a factor whose per-bar "
                        "turnover (mean |Δposition|, ∈[0,2]) exceeds this floor. "
                        "Default off — turnover/net-cost are still reported as "
                        "diagnostics regardless.")
    p.add_argument("--cost-rate", type=float, default=5e-4,
                   help="Per-unit-turnover cost (≈5 bps) for the net-of-cost "
                        "diagnostics (default 0.0005).")
    p.add_argument("--perturbation-weight", type=float, default=0.0,
                   help="Weight of the marginal-axis perturbation probe (0 = off, the "
                        "baseline arm). >0 docks the marginal-value axis by the "
                        "sign-aligned VAL-IC drop under a Gaussian signal shock.")
    p.add_argument("--perturbation-sigma", type=float, default=0.5,
                   help="Stdev (in signal z-units) of the perturbation shock "
                        "(default 0.5).")

    # ── two-stage curation (Lever 2) ──
    p.add_argument("--curation", choices=["archive", "greedy", "elastic_net"],
                   default="archive",
                   help="How the final book is chosen. 'archive' (default): the "
                        "Pareto archive (one-stage, old behaviour). 'greedy' / "
                        "'elastic_net': keep every gate-passing factor during the "
                        "search, then curate the pool once at the end (two-stage).")
    p.add_argument("--n-keep", type=int, default=None,
                   help="Target number of factors to keep at curation "
                        "(default: auto-sized — greedy stops on no marginal gain, "
                        "elastic-net keeps those above the stability threshold).")
    # ── selection-time deflation (WS1): multiple-testing honesty, moved off the
    # per-candidate search gate onto the final published book ──
    p.add_argument("--selection-deflation", choices=["off", "on"], default="off",
                   help="Deflate the final book's COMBINED OOS IC for the run's "
                        "N_trials at publish. 'off' (default): discovery mode — keep "
                        "everything, deflation reported not enforced. 'on': validation "
                        "mode — narrow the book (pruning by marginal contribution) to "
                        "what beats selection luck.")
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
    from quant_fund_agent.research_eval.prebook import book_entries, load_prebook

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
    fixed_book = book_entries(load_prebook(args.fixed_book)) if args.fixed_book else []
    if fixed_book:
        log.info("Conditioning fitness on fixed book: %d factor(s) from %s",
                 len(fixed_book), args.fixed_book)
    reference_book = (book_entries(load_prebook(args.reference_book))
                      if args.reference_book else [])
    if reference_book:
        log.info("Novelty diagnostic vs reference book: %d factor(s) from %s",
                 len(reference_book), args.reference_book)

    cfg = EvolutionRunConfig(
        generations=args.generations,
        population_size=args.population,
        children_per_generation=args.children_per_gen or 8,
        children_per_deme=(None if args.children_per_gen is not None
                           else args.children_per_deme),
        n_mechanism_groups=args.mechanism_groups,
        demes_per_group=(args.islands if args.islands is not None
                         else args.demes_per_group),
        migration_every=args.migration_every,
        seed=args.seed,
        unit=args.evolution_unit,
        set_size=args.set_size,
        p_llm_semantic=args.p_llm,
        p_crossover=args.p_crossover,
        p_cross_group=args.p_cross_group,
        p_jitter=args.p_jitter,
        n_seed_ideas=args.seed_ideas_per_group,
        seed_ideas_per_group=args.seed_ideas_per_group,
        seed_papers=args.seed_papers,
        retrieval=args.retrieval,
        retrieval_cardinality=args.retrieval_cardinality,
        rag_k=args.rag_k,
        debate=args.debate,
        target_horizon=args.horizon,
        is_frac=args.is_frac,
        val_frac=args.val_frac,
        stability_blocks=args.stability_blocks,
        cutoff_date=args.cutoff_date,
        progressive=args.progressive_reveal,
        test_frac=args.test_frac,
        seed_frac=args.seed_frac,
        reveal_every=args.reveal_every,
        val_blocks=args.val_blocks,
        force_prediction_horizon=(args.prediction_horizon_mode == "fixed"),
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
        fixed_book=fixed_book,
        reference_book=reference_book,
        memory=args.memory,
        memory_config=config_name,
        out_dir=str(scope.dir / "evolution"),
    )

    if cfg.progressive:
        log.info("progressive reveal ON (test_frac=%.2f, seed_frac=%.2f, "
                 "reveal_every=%d, val_blocks=%d) — --is-frac/--val-frac are ignored; "
                 "the split follows the per-generation calendar schedule.",
                 cfg.test_frac, cfg.seed_frac, cfg.reveal_every, cfg.val_blocks)

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
        data_dir=args.data_dir, n_tickers=args.n_tickers,
        curation=args.curation, n_keep=args.n_keep,
        is_frac=args.is_frac, val_frac=args.val_frac, fields=loop.fields,
        marginal_model=args.marginal_model,
        selection_deflation=args.selection_deflation)
    summary["persisted_factor_ids"] = persisted["kept_factor_ids"]
    summary["curation"] = args.curation
    summary["selection_deflation"] = args.selection_deflation

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
