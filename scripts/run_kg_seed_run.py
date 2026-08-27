"""One seeding-only research run for the evolving knowledge-graph breadth campaign.

``--run-index N`` performs ONE generation-0 style seeding pass against the LIVE
knowledge graph (``data/knowledge/graph.json`` — deliberately not pinned via
``QF_GRAPH_PATH``: coverage shifts run-over-run are the point) with NO fitness
evaluation, NO controller and NO curation:

1. mechanism groups are re-resolved from the current graph every run
   (``resolve_mechanism_groups``, graphrag, ``mechanism_groups_mode="max"``);
2. each group is seeded through the loop's existing ``seed_programs`` path
   (retrieval-grounded hypothesis → codegen → in-memory validation);
3. deterministic dedup only — ids already known to the package or the
   campaign's cumulative book, plus EXACT canonical-AST structural clones of a
   cumulative-book member (``research_eval.ast_novelty`` fingerprints; this is
   a clone check, not a novelty score);
4. survivors are persisted WITHOUT evaluation (factor file + a minimal
   ``factor_db.json`` record in the run's prerun) and linked back into the
   live graph (``link_programs_into_graph(..., readonly=False)``), so the
   NEXT run resolves its mechanism groups from a shifted graph.

Note the link-back reuses the loop's own helper verbatim, which rebuilds the
graph's ``uses`` edges from THIS run's programs only (the same behaviour a live
non-readonly evolution run has); ``scripts/link_factors_into_graph.py
--merge-uses-from`` repairs older ``uses`` edges if that ever matters.

Exit codes: 0 ok · 3 zero factors persisted · 4 LLM budget ceiling hit
(everything made before the ceiling is still persisted).

Usage:
    ./venv/bin/python scripts/run_kg_seed_run.py --run-index 1
    ./venv/bin/python scripts/run_kg_seed_run.py --run-index 2 --max-cost-usd 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_kg_seed_run")

DEFAULT_CAMPAIGN_DIR = "data/kg_campaign"
DEFAULT_WORKSPACE_ROOT = "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-index", type=int, required=True,
                   help="Campaign run number N (prerun KG<NN>_terra_s0; also "
                        "the seeding RNG/session seed).")
    p.add_argument("--model", default="gpt-5.6-terra",
                   help="Research LLM model id (sets FACTOR_RESEARCH_LLM_MODEL).")
    p.add_argument("--llm-provider", default="openai",
                   help="Sets FACTOR_RESEARCH_LLM_PROVIDER.")
    p.add_argument("--max-cost-usd", type=float, default=15.0,
                   help="Hard ceiling on cumulative LLM spend (sets "
                        "QF_MAX_LLM_COST_USD; seeding stops when crossed).")
    p.add_argument("--seed-ideas-per-group", type=int, default=12,
                   help="Brainstorm budget for every resolved mechanism group.")
    p.add_argument("--mechanism-groups", type=int, default=8,
                   help="Upper limit on knowledge-graph mechanism groups "
                        "(mechanism_groups_mode='max').")
    p.add_argument("--horizon", type=int, default=6,
                   help="Forced prediction_horizon (bars) for every seeded factor.")
    p.add_argument("--config", "--config-file", dest="config", default=None,
                   help="Path to a quant.config.<x>.yaml (sets QF_CONFIG_FILE "
                        "before the field scope is resolved).")
    p.add_argument("--data-dir", default=os.getenv("DATA_DIR", "ticker_data"))
    p.add_argument("--campaign-dir", default=DEFAULT_CAMPAIGN_DIR,
                   help="Campaign state dir (cumulative_book.json + summaries).")
    p.add_argument("--workspace-root", default=DEFAULT_WORKSPACE_ROOT,
                   help="Preruns root the KG<NN>_terra_s0 factor DB is written "
                        "under.")
    return p.parse_args(argv)


# ── helpers (module-level so tests can monkeypatch them) ──────────────────────

def _data_context_and_fields(cfg) -> tuple[str, list[str] | None]:
    """The seeding DATA CONTEXT, built exactly like ``EvolutionLoop._build_data_context``."""
    from quant_fund_agent.agents.factor_research.prompts import build_data_context

    allowed: list[str] | None = None
    try:
        from quant_fund_agent.data import usable_fields

        allowed = sorted(usable_fields())
    except Exception as e:  # noqa: BLE001 — mirror the loop: un-gated fallback
        log.warning("could not resolve usable fields (%s) — un-gated", e)
    spb = None
    try:
        from quant_fund_agent.pipeline import _infer_seconds_per_bar

        spb = _infer_seconds_per_bar(cfg.data_dir)
    except Exception:  # noqa: BLE001
        pass
    fixed = cfg.target_horizon if cfg.force_prediction_horizon else None
    return build_data_context(allowed, spb, fixed_prediction_horizon=fixed), allowed


def _group_context(data_context: str, spec: dict) -> str:
    """Per-group seeding context (mirrors ``EvolutionLoop._group_context``)."""
    focus = str(spec.get("focus") or "").strip()
    if not focus:
        return data_context
    return (
        data_context
        + "\n\nKNOWLEDGE-GRAPH MECHANISM GROUP\n"
        + "Develop this lineage within the following economic mechanism "
          "community. You may combine related mechanisms, but keep a clear "
          "causal link to the group:\n"
        + focus
    )


def _canonical_fp(code: str) -> str | None:
    """Deterministic canonical-AST fingerprint for the EXACT-clone dedup check.

    Two codes share a fingerprint iff their canonical subtree profiles are
    equal (same canonical computation — ids/names/windows normalised away).
    ``None`` for empty/invalid source → the program is kept (clone check only,
    never a fabricated drop).
    """
    from quant_fund_agent.research_eval.ast_novelty import subtree_profile

    prof = subtree_profile(code)
    if prof is None:
        return None
    return hashlib.sha256(repr(prof.items).encode("utf-8")).hexdigest()


def _factor_record(prog, code_path: str, run_index: int,
                   model: str, provider: str | None) -> dict:
    """Minimal factor-DB row (the shape of an existing prerun record, no metrics)."""
    from quant_fund_agent.factors.inmem import compile_factor
    from quant_fund_agent.schemas import TradingIdeaCategory

    inputs = ["close"]
    try:
        cls = compile_factor(prog.code, prog.factor_id)
        inputs = list(getattr(cls, "inputs", None) or ["close"])
    except Exception:  # noqa: BLE001 — the program already validated at seeding
        pass
    try:
        category = TradingIdeaCategory((prog.category or "other").lower()).value
    except ValueError:
        category = TradingIdeaCategory.OTHER.value
    return {
        "id": prog.factor_id,
        "name": prog.name or prog.factor_id,
        "description": prog.description,
        "trading_idea": prog.trading_idea,
        "category": category,
        "status": "candidate",
        "source": "researcher",
        "research_session_id": f"kg-campaign:{run_index:02d}",
        "code_path": code_path,
        "required_inputs": inputs,
        "inputs": inputs,
        "prediction_horizon": int(prog.prediction_horizon or 6),
        "suggested_horizons": list(prog.suggested_horizons or []),
        "source_paper_ids": list(prog.source_paper_ids or []),
        "created_at": datetime.utcnow().isoformat(),
        "metadata": {
            "kg_campaign_run": run_index,
            "mechanism": prog.mechanism or "",
            "mechanism_group_id": int(prog.mechanism_group_id),
            "engine": "kg_seed",
            "llm_model": model,
            "llm_provider": provider or "",
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # LLM env wiring, exactly like run_factor_evolution.py's main().
    if args.config:
        os.environ["QF_CONFIG_FILE"] = args.config
    os.environ["FACTOR_RESEARCH_LLM_MODEL"] = args.model
    os.environ["FACTOR_RESEARCH_LLM_PROVIDER"] = args.llm_provider
    os.environ["QF_MAX_LLM_COST_USD"] = str(args.max_cost_usd)
    os.environ["DATA_DIR"] = args.data_dir
    # No evaluation happens in this entrypoint, so there is no reason to spawn
    # the MCP research server for the one existing_factor_ids() lookup.
    os.environ.setdefault("QF_USE_MCP", "0")

    from quant_fund_agent.agents.factor_research import codegen
    from quant_fund_agent.agents.factor_research.evolution import loop as evolution_loop
    from quant_fund_agent.llm import (
        LLMBudgetExceeded,
        budget_exhausted,
        resolve_research_model,
        resolve_research_provider,
        usage_summary,
    )
    from quant_fund_agent.mcp import research_client

    model = resolve_research_model()
    provider = resolve_research_provider(model)
    if provider in (None, "openai") and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment first.")
    log.info("KG campaign run %d (model=%s, provider=%s, ceiling=$%.2f)",
             args.run_index, model, provider, args.max_cost_usd)

    # Seeding-only config: graphrag groups from the LIVE graph.  graph_readonly
    # here only silences seed_programs' per-group link-back — the one final
    # link-back below (readonly=False) writes the DEDUPED persisted set, so
    # clones and dropped ids never enter the graph.
    cfg = evolution_loop.EvolutionRunConfig(
        retrieval="graphrag",
        n_mechanism_groups=args.mechanism_groups,
        mechanism_groups_mode="max",
        n_seed_ideas=args.seed_ideas_per_group,
        seed_ideas_per_group=args.seed_ideas_per_group,
        seed=args.run_index,
        target_horizon=args.horizon,
        force_prediction_horizon=True,
        data_dir=args.data_dir,
        graph_readonly=True,
    )

    data_context, fields = _data_context_and_fields(cfg)
    groups = evolution_loop.resolve_mechanism_groups(cfg, fields)
    log.info("resolved %d mechanism group(s) from the live graph", len(groups))

    # ── campaign state: cumulative book (ids + canonical fingerprints) ──
    campaign_dir = Path(args.campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    book_path = campaign_dir / "cumulative_book.json"
    book: list[dict] = (json.loads(book_path.read_text())
                        if book_path.exists() else [])
    book_ids = {str(row.get("factor_id")) for row in book}
    book_fps = {row["canonical_fp"] for row in book if row.get("canonical_fp")}

    known: set[str] = set()
    try:
        known = set(research_client.existing_factor_ids(scope="package"))
    except Exception as e:  # noqa: BLE001 — mirror the loop's fail-open lookup
        log.warning("could not load existing factor ids (%s)", e)
    known |= book_ids

    # ── seeding: one seed_programs pass per resolved group ──
    programs: list = []
    budget_hit = False
    try:
        for spec in groups:
            if budget_exhausted():
                budget_hit = True
                log.warning("LLM budget ceiling reached — stopping before "
                            "group %s", spec.get("mechanism_group_id"))
                break
            group_programs = evolution_loop.seed_programs(
                cfg, _group_context(data_context, spec), known,
                fields=fields, mechanism_group=spec)
            known.update(p.factor_id for p in group_programs)
            programs.extend(group_programs)
    except LLMBudgetExceeded as e:
        # Never discard paid work: fall through and persist what was made.
        budget_hit = True
        log.warning("LLM budget ceiling reached mid-seeding (%s) — persisting "
                    "what was made", e)

    # ── deterministic dedup: known ids + exact canonical-AST clones ──
    kept: list[tuple] = []          # (program, canonical_fp)
    n_deduped = 0
    seen_fps = set(book_fps)
    for prog in programs:
        if prog.factor_id in book_ids:   # belt-and-braces; seed_programs skips known
            log.info("dropping %s: id already in the cumulative book",
                     prog.factor_id)
            n_deduped += 1
            continue
        fp = _canonical_fp(prog.code)
        if fp is not None and fp in seen_fps:
            log.info("dropping %s: exact structural clone of an existing "
                     "cumulative-book member", prog.factor_id)
            n_deduped += 1
            continue
        if fp is not None:
            seen_fps.add(fp)
        kept.append((prog, fp))

    # ── persist WITHOUT evaluation: factor file + minimal prerun DB record ──
    prerun_factors_dir = (Path(args.workspace_root)
                          / f"KG{args.run_index:02d}_terra_s0" / "factors")
    prerun_factors_dir.mkdir(parents=True, exist_ok=True)
    db_path = prerun_factors_dir / "factor_db.json"
    db = (json.loads(db_path.read_text()) if db_path.exists()
          else {"factors": [], "trading_ideas": []})
    db.setdefault("factors", [])
    db_ids = {row.get("id") for row in db["factors"]}

    for prog, fp in kept:
        code_path = codegen.write_factor_file(prog.factor_id, prog.code)
        record = _factor_record(prog, str(code_path), args.run_index,
                                model, provider)
        if prog.factor_id not in db_ids:
            db["factors"].append(record)
            db_ids.add(prog.factor_id)
        book.append({
            "factor_id": prog.factor_id,
            "run": args.run_index,
            "code_path": str(code_path),
            "canonical_fp": fp,
        })
    db_path.write_text(json.dumps(db, indent=2))
    book_path.write_text(json.dumps(book, indent=2))
    log.info("persisted %d factor(s) into %s", len(kept), db_path)

    # ── graph link-back of the persisted set (this is the campaign's point:
    # the next run resolves its groups from the shifted graph) ──
    kept_programs = [prog for prog, _fp in kept]
    if kept_programs:
        from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

        mech_by_fid = {p.factor_id: p.mechanism
                       for p in kept_programs if p.mechanism}
        try:
            graph = KnowledgeGraph.load()
            # saves graph.json itself (same call path a live seeding run uses)
            evolution_loop.link_programs_into_graph(
                graph, kept_programs, mech_by_fid, readonly=False)
        except Exception as e:  # noqa: BLE001 — provenance is best-effort
            log.warning("graph link-back failed (%s)", e)

    # ── run summary (timestamp-free: byte-identical given identical inputs) ──
    summary = {
        "run": args.run_index,
        "n_ideas_requested": len(groups) * args.seed_ideas_per_group,
        "n_validated": len(programs),
        "n_deduped": n_deduped,
        "n_persisted": len(kept),
        "llm_cost_usd": usage_summary().get("total", {}).get("cost_usd", 0.0),
        "group_ids": [int(s["mechanism_group_id"]) for s in groups],
    }
    summary_path = campaign_dir / f"run_{args.run_index:02d}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if budget_hit or budget_exhausted():
        log.warning("run stopped on the LLM budget ceiling — persisted %d "
                    "factor(s); exiting 4", len(kept))
        return 4
    if not kept:
        log.error("run persisted ZERO factors — exiting 3")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
