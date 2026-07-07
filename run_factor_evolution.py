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
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--population", type=int, default=10, help="Per-island size.")
    p.add_argument("--children-per-gen", type=int, default=8)
    p.add_argument("--islands", type=int, default=1)
    p.add_argument("--migration-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the controller.")
    # ── selection: NSGA-II Pareto vs QD behavior grid (WS2) ──
    p.add_argument("--selection", choices=["nsga2", "qd"], default="nsga2",
                   help="Parent-selection / archive mode. 'nsga2' (default): the "
                        "Pareto archive drives selection. 'qd': a MAP-Elites behavior "
                        "grid (trend_reversal × signal_speed) fills a diverse library "
                        "while keeping the SAME 5-axis Pareto as per-cell quality.")
    p.add_argument("--grid-dims", type=int, choices=[2, 3], default=2,
                   help="QD behavior-grid dimensionality: 2 (trend×speed, default) or "
                        "3 (+ stress_activation).")
    p.add_argument("--cell-capacity", type=int, default=3,
                   help="QD mini-Pareto elites kept per behavior cell (default 3).")
    p.add_argument("--depth-gamma", type=float, default=0.0,
                   help="P7 depth penalty (1-γ)^depth on QD parent sampling "
                        "(0 = off; AlphaPROBE-style anti-overfit bias).")
    p.add_argument("--reuse-omega", type=float, default=0.0,
                   help="P7 parent-reuse penalty (1-ω)^reuse on QD parent sampling "
                        "(0 = off; anti mode-collapse bias).")

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
    p.add_argument("--cpcv-model", default=None,
                   help="Estimator used for fold-refit CPCV robustness. Default: "
                        "reuse --marginal-model. Keep this nonlinear "
                        "(e.g. gradient_boosting/lightgbm) if you want "
                        "conditioning factors to pass robustness.")
    p.add_argument("--no-cpcv-fast", action="store_true",
                   help="Disable the lightweight tree/boosting preset for repeated "
                        "CPCV refits. By default CPCV refits use fast model params "
                        "because they run once per fold.")
    p.add_argument("--prediction-horizon-mode", choices=["fixed", "free"],
                   default="fixed",
                   help="fixed (default): force every generated factor to declare "
                        "--horizon as prediction_horizon. free: let the researcher "
                        "declare per-factor horizons, while fitness still scores at "
                        "--horizon.")
    p.add_argument("--cutoff-date", default=None,
                   help="ISO date: evaluate only bars strictly before this "
                        "(walk-forward wrapping).")
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
    p.add_argument("--regime-kind", choices=["drawdown", "volatility"],
                   default="drawdown",
                   help="Stress bars for the regime axis: worst market-return "
                        "'drawdown' bars (default) or top-volatility bars.")
    p.add_argument("--regime-quantile", type=float, default=0.2,
                   help="Tail fraction of dev bars labelled 'stress' (default 0.2).")
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
                   help="Weight of the perturbation-fidelity probe folded into the "
                        "robustness axis (0 = off, the baseline arm). >0 docks "
                        "robustness by the sign-aligned VAL-IC drop under a Gaussian "
                        "signal shock.")
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
        children_per_generation=args.children_per_gen,
        n_islands=args.islands,
        migration_every=args.migration_every,
        seed=args.seed,
        selection=args.selection,
        grid_dims=args.grid_dims,
        cell_capacity=args.cell_capacity,
        depth_gamma=args.depth_gamma,
        reuse_omega=args.reuse_omega,
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
        cpcv_model=args.cpcv_model,
        cpcv_fast=not args.no_cpcv_fast,
        cutoff_date=args.cutoff_date,
        force_prediction_horizon=(args.prediction_horizon_mode == "fixed"),
        independence_metric=args.independence_metric,
        regime_kind=args.regime_kind,
        regime_quantile=args.regime_quantile,
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
