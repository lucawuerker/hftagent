"""Factor-research **prerun**: mine a batch of alpha factors once and keep them.

Research is strictly modularised by **(config, prerun)** and ALWAYS persists into a
scope under ``data/workspaces/<config>/preruns/<prerun>/`` (its own
``factor_db.json`` + ``manifest.json`` + paper read-log) — never the seed-only
*main* library ``data/factors/factor_db.json``.  The config scope defaults to the
active ``quant.config.yaml`` (e.g. ``yfinance_equity_demo``); the prerun defaults
to ``base``.

Use ``--name`` to compare research LLMs: mine ~100 factors with one ``--model``
into prerun A and ~100 with another into prerun B, then run the downstream agents
on each (``run_fund.py --prerun A`` / ``run_backtest.py --prerun A``) and compare.
The generated ``.py`` code stays in the shared ``factors/researcher/`` package
(global id de-dup); the scope owns its factor DB.

Paper read-tracking is its own scenario per scope (a read-log), so successive
sessions advance through the paper library without biasing any backtest's paper
selection.  The loop keeps researching (broad paper sample, ~1 idea per paper)
until ``--target-factors`` are kept or ``--max-sessions``.

Examples
--------
::

    # The active config's 'base' prerun: ~100 factors.
    python run_factor_research.py --target-factors 100

    # Named prerun with a specific research model (model A).
    python run_factor_research.py --name gpt4omini --model gpt-4o-mini --target-factors 100

    # Model B (non-OpenAI needs e.g. `pip install langchain-anthropic` + key).
    python run_factor_research.py --name claude --model claude-3-5-sonnet-latest \
      --llm-provider anthropic --target-factors 100

    # Smoke run, in-process, tiny; start a named prerun over from scratch.
    QF_USE_MCP=0 python run_factor_research.py --name smokeA --reset \
      --target-factors 2 --papers-per-session 2 --ideas-per-session 2 \
      --max-sessions 1 --n-tickers 8
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_factor_research")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default=None,
                   help="Prerun id (research-LLM batch) within the config scope → "
                        "data/workspaces/<config>/preruns/<name>/.  Omit → the "
                        "'base' prerun.  Research never writes the seed-only main DB.")
    p.add_argument("--config-name", default=None,
                   help="Config scope name under data/workspaces/<name>/ "
                        "(default: derived from the active config, e.g. "
                        "yfinance_equity_demo).")
    p.add_argument("--model", default=None,
                   help="Research LLM model id (sets FACTOR_RESEARCH_LLM_MODEL).")
    p.add_argument("--llm-provider", default=None,
                   help="LLM provider (openai|anthropic|google_genai|…). "
                        "Inferred from the model id when omitted.")
    p.add_argument("--target-factors", type=int, default=100,
                   help="Stop once this prerun holds this many researcher factors.")
    p.add_argument("--papers-per-session", type=int, default=25)
    p.add_argument("--ideas-per-session", type=int, default=25,
                   help="Total idea budget per session (split across the papers).")
    p.add_argument("--max-sessions", type=int, default=12,
                   help="Hard cap on the number of research sessions.")
    p.add_argument("--horizon", type=int, default=6,
                   help="Run-level forecast horizon in bars. In fixed-horizon mode, "
                        "every generated factor must declare this as its "
                        "prediction_horizon; IC is recorded here and not gated.")
    p.add_argument("--prediction-horizon-mode", choices=["fixed", "free"],
                   default="fixed",
                   help="fixed (default): force every researched factor to use "
                        "--horizon as prediction_horizon. free: let the researcher "
                        "declare per-factor horizons while still recording IC at "
                        "--horizon.")
    p.add_argument("--n-tickers", type=int, default=15,
                   help="Universe cap for the research-time IC backtest.")
    p.add_argument("--sample", choices=["random", "unread_first"], default="random",
                   help="How to pick each session's papers from the unread pool.")
    p.add_argument("--dedup-scope", choices=["package", "prerun"], default="package",
                   help="De-dup new factor ids against all shared-package code "
                        "('package', default) or only THIS prerun's factors "
                        "('prerun', so the brainstorm isn't anchored by other "
                        "preruns — fairer for an A/B of research models).")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--reset", action="store_true",
                   help="Purge this prerun's factors + code + read-log before starting "
                        "(global researcher library if --name is absent).")
    return p.parse_args()


def _researcher_count(db_path: Path) -> int:
    """How many RESEARCHER factors are currently in ``db_path``."""
    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    db = FactorDatabase()
    db.load_from_json(db_path)
    return len(db.list_factors(source=FactorSource.RESEARCHER))


def main() -> None:
    args = _parse_args()
    # 0 / negative --n-tickers = the full universe (n_tickers=None is the
    # no-cap convention downstream; 0 would slice the universe to nothing).
    args.n_tickers = args.n_tickers if args.n_tickers > 0 else None
    if not os.getenv("OPENAI_API_KEY") and (args.llm_provider in (None, "openai")):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")

    # ── research LLM: set BEFORE importing the research graph so the model id
    #    (used by the LLM factory + the tiktoken encoder) is the chosen one.
    if args.model:
        os.environ["FACTOR_RESEARCH_LLM_MODEL"] = args.model
    if args.llm_provider:
        os.environ["FACTOR_RESEARCH_LLM_PROVIDER"] = args.llm_provider
    os.environ["DATA_DIR"] = args.data_dir

    from quant_fund_agent import pipeline
    from quant_fund_agent.config import default_config_name, get_settings
    from quant_fund_agent.llm import resolve_research_model, resolve_research_provider
    from quant_fund_agent.workspace import Scope

    # ── resolve the (config, prerun) scope research persists into ──
    #   Research ALWAYS writes a scope's researcher-only DB — never the seed-only
    #   main library — so factors stay strictly modularised by (config, prerun).
    settings = get_settings()
    config_name = args.config_name or default_config_name(settings.data)
    prerun = args.name or "base"
    scope = Scope(config_name, prerun)
    scope.ensure()
    scope.write_config_snapshot(settings.data)
    factor_db_path = scope.factor_db_path
    read_log = scope.read_log_path
    os.environ["FACTOR_DB_PATH"] = str(factor_db_path)
    os.environ["PAPER_READ_LOG"] = str(read_log)
    os.environ["QF_SCOPE"] = scope.label

    if args.reset:
        log.info("Resetting scope '%s' (factors, code, read-log) …", scope.label)
        scope.purge()
        scope.ensure()

    model = resolve_research_model()
    provider = resolve_research_provider(model)
    label = f"scope '{scope.label}'"
    log.info("Researching into %s at %s (model=%s, provider=%s); target %d",
             label, factor_db_path, model, provider, args.target_factors)

    for i in range(1, args.max_sessions + 1):
        have = _researcher_count(factor_db_path) if factor_db_path.exists() else 0
        if have >= args.target_factors:
            log.info("Target reached (%d ≥ %d) — stopping.", have, args.target_factors)
            break
        log.info("── session %d/%d  (have %d, want %d) ──",
                 i, args.max_sessions, have, args.target_factors)
        sid = f"prerun:{config_name}:{prerun}:{i:03d}"
        try:
            rs = pipeline.run_research_session(
                session_id=sid,
                cutoff_date=None,                       # whole library (per decision)
                n_papers=args.papers_per_session,
                n_ideas=args.ideas_per_session,
                horizon=args.horizon,
                force_prediction_horizon=(args.prediction_horizon_mode == "fixed"),
                n_tickers=args.n_tickers,
                paper_sample_strategy=args.sample,
                data_dir=args.data_dir,
                dedup_scope=args.dedup_scope,
            )
            kept = rs.get("kept_factor_ids", [])
            log.info("session %d kept %d factor(s): %s", i, len(kept), kept)
            scope.write_manifest(
                llm_model=model, llm_provider=provider,
                target_factors=args.target_factors,
                n_factors=_researcher_count(factor_db_path),
                sessions=i,
            )
            if not rs.get("selected_papers"):
                log.warning("No papers left to read — stopping early.")
                break
        except Exception as e:  # noqa: BLE001 — one bad session must not abort the prerun
            log.warning("session %d failed: %s", i, e)

    _summarise(factor_db_path, label, model)


def _summarise(factor_db_path: Path, label: str, model: str) -> None:
    from collections import Counter

    from quant_fund_agent.databases import FactorDatabase
    from quant_fund_agent.schemas import FactorSource

    db = FactorDatabase()
    db.load_from_json(factor_db_path)
    researcher = db.list_factors(source=FactorSource.RESEARCHER)
    by_cat = Counter(f.category.value for f in researcher)

    print("\n" + "=" * 80)
    print(f"Factor {label}: {factor_db_path}  (research model: {model})")
    print(f"  seed factors      : {len(db.list_factors(source=FactorSource.SEED))}")
    print(f"  researcher factors: {len(researcher)}")
    print(f"  by category       : {dict(by_cat)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
