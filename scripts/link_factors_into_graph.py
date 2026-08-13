"""Post-hoc provenance link-back: completed runs' factor books → knowledge graph.

The WF ladder ran ``--graph-readonly`` (every arm ranks mechanism groups from
the identical snapshot), so none of its surviving factors were ever linked
back into ``data/knowledge/graph.json``.  This script closes the
Paper → Mechanism → Factor → Field provenance loop after the fact, additively:

* factor nodes for every *published* book member of the given preruns
  (node attrs ``preruns`` = which runs produced it, ``engine``);
* ``realized_by`` edges where a mechanism is known — either stamped on the
  program (seeds) or inherited by walking the genome parent chain back to the
  nearest ancestor that carries one (edge attr ``provenance`` =
  ``seed`` / ``inherited``);
* ``uses`` edges from each factor's declared ``required_inputs`` (factor-DB
  record, falling back to compiling the code).  NEVER via
  ``refresh_field_usage`` — that helper drops ALL existing ``uses`` edges and
  rebuilds only from its argument, which is how the local graph lost the
  ladder snapshot's 71 ``uses`` edges in the first place.
* optional ``--merge-uses-from <snapshot>``: re-adds ``uses`` edges present in
  an older graph file but missing here (repairs exactly that clobbering);
* a final ``refresh_mechanism_coverage`` so the coverage stats
  (``n_factors`` / ``mean_abs_ic``) future gap queries rank by are current.

Usage:
    ./venv/bin/python scripts/link_factors_into_graph.py \
        --prerun L4WF_terra_s0=/path/to/prerun_dir [...] \
        [--merge-uses-from data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json] \
        [--dry-run]

A prerun dir needs ``factors/factor_db.json`` and, for evolution arms,
``evolution/state.json`` (or ``gp/state.json``).  Oneshot arms (no state) get
factor nodes + ``uses`` edges only — their records carry no mechanism.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_fund_agent.knowledge.empirical_edges import (  # noqa: E402
    link_factor_to_mechanism,
    refresh_mechanism_coverage,
)
from quant_fund_agent.knowledge.graph_store import KnowledgeGraph  # noqa: E402

log = logging.getLogger("link_factors_into_graph")


# ── prerun readers ─────────────────────────────────────────────────────────

def _load_factor_db(prerun_dir: Path) -> dict[str, dict]:
    p = prerun_dir / "factors" / "factor_db.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    recs = raw.get("factors", raw)
    if isinstance(recs, dict):
        recs = list(recs.values())
    return {r["id"]: r for r in recs if isinstance(r, dict) and r.get("id")}


def _iter_genome_entries(obj) -> list[dict]:
    """Every genome dict reachable in a state.json fragment (archive entries,
    kept_pool entries, nested island lists)."""
    out: list[dict] = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "genome_id" in cur and "programs" in cur:
                out.append(cur)
            else:
                stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return out


def _load_state(prerun_dir: Path) -> dict | None:
    for sub in ("evolution", "gp"):
        p = prerun_dir / sub / "state.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def _resolve_mechanism(genome: dict, mech_of: dict[str, str],
                       parents_of: dict[str, list[str]]) -> tuple[str, str] | None:
    """(mechanism, provenance) for a genome — its own stamp, else the nearest
    ancestor's (BFS over parent_ids)."""
    own = next((p.get("mechanism") for p in genome.get("programs", [])
                if p.get("mechanism")), None)
    if own:
        return own, "seed"
    seen: set[str] = set()
    queue = deque(genome.get("parent_ids") or [])
    while queue:
        gid = queue.popleft()
        if gid in seen:
            continue
        seen.add(gid)
        mech = mech_of.get(gid)
        if mech:
            return mech, "inherited"
        queue.extend(parents_of.get(gid, []))
    return None


def collect_book(prerun_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """The prerun's published book as
    ``[{factor_id, code, mechanism, provenance, group}, …]`` + its factor-DB
    records by id.  Evolution arms: final Pareto archive (deduped by factor
    id, restricted to published DB ids when a DB exists); oneshot arms: every
    researcher DB record."""
    db = _load_factor_db(prerun_dir)
    state = _load_state(prerun_dir)
    if state is None:
        book = [{"factor_id": fid, "code": None, "mechanism": None,
                 "provenance": None, "group": None}
                for fid, r in db.items() if r.get("source") in (None, "researcher")]
        return book, db

    all_genomes = _iter_genome_entries(
        [state.get("archive"), state.get("kept_pool"), state.get("islands")])
    mech_of: dict[str, str] = {}
    parents_of: dict[str, list[str]] = {}
    for g in all_genomes:
        parents_of.setdefault(g["genome_id"], g.get("parent_ids") or [])
        mech = next((p.get("mechanism") for p in g.get("programs", [])
                     if p.get("mechanism")), None)
        if mech:
            mech_of.setdefault(g["genome_id"], mech)

    book: list[dict] = []
    seen_fids: set[str] = set()
    for entry in state.get("archive", []):
        genome = entry.get("genome", entry)
        resolved = _resolve_mechanism(genome, mech_of, parents_of)
        for prog in genome.get("programs", []):
            fid = prog.get("factor_id")
            if not fid or fid in seen_fids:
                continue
            if db and fid not in db:      # archived but not published
                continue
            seen_fids.add(fid)
            mech, prov = (prog.get("mechanism"), "seed") if prog.get("mechanism") \
                else (resolved if resolved else (None, None))
            book.append({"factor_id": fid, "code": prog.get("code"),
                         "mechanism": mech, "provenance": prov,
                         "group": genome.get("mechanism_group_id")})
    return book, db


# ── graph writers ──────────────────────────────────────────────────────────

def _inputs_for(fid: str, code: str | None, db: dict[str, dict]) -> list[str]:
    rec = db.get(fid)
    if rec and rec.get("required_inputs"):
        return list(rec["required_inputs"])
    if code:
        try:
            from quant_fund_agent.factors.inmem import compile_factor
            cls = compile_factor(code, fid)
            return list(getattr(cls, "inputs", None) or [])
        except Exception:  # noqa: BLE001 — provenance is best-effort
            pass
    return []


def merge_uses_edges(graph: KnowledgeGraph, snapshot_path: Path) -> int:
    snap = json.loads(snapshot_path.read_text())
    n = 0
    for e in snap.get("edges", []):
        if e.get("relation") != "uses":
            continue
        src, dst = e["source"], e["target"]
        if graph.g.has_edge(src, dst):
            continue
        if src.startswith("factor:"):
            graph.add_factor(src.split(":", 1)[1])
        if dst.startswith("field:"):
            graph.add_field(dst.split(":", 1)[1])
        graph.add_relation(src, dst, "uses")
        n += 1
    return n


def link_prerun(graph: KnowledgeGraph, name: str, prerun_dir: Path,
                ic_map: dict[str, float]) -> dict[str, int]:
    book, db = collect_book(prerun_dir)
    stats = {"factors": 0, "realized_by": 0, "inherited": 0, "uses": 0,
             "no_mechanism": 0}
    for item in book:
        fid = item["factor_id"]
        rec = db.get(fid, {})
        node = graph.add_factor(fid, name=rec.get("name", fid),
                                category=rec.get("category", ""))
        runs = set(graph.g.nodes[node].get("preruns") or [])
        runs.add(name)
        graph.g.nodes[node]["preruns"] = sorted(runs)
        engine = (rec.get("metadata") or {}).get("engine")
        if engine:
            graph.g.nodes[node]["engine"] = engine
        stats["factors"] += 1

        if item["mechanism"]:
            link_factor_to_mechanism(graph, fid, item["mechanism"])
            mid = graph.mechanism_id(item["mechanism"])
            graph.g.edges[mid, node]["provenance"] = item["provenance"]
            stats["realized_by"] += 1
            if item["provenance"] == "inherited":
                stats["inherited"] += 1
        else:
            stats["no_mechanism"] += 1

        for field in _inputs_for(fid, item["code"], db):
            if not graph.g.has_edge(node, graph.field_id(field)):
                graph.add_relation(node, graph.add_field(field), "uses")
                stats["uses"] += 1

        ic = ((rec.get("backtest_metrics") or {}).get("information_coefficient"))
        if ic is not None and fid not in ic_map:
            ic_map[fid] = float(ic)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", default="data/knowledge/graph.json")
    ap.add_argument("--out", default=None,
                    help="output path (default: overwrite --graph)")
    ap.add_argument("--merge-uses-from", default=None,
                    help="older graph.json whose 'uses' edges are re-added")
    ap.add_argument("--prerun", action="append", default=[],
                    metavar="NAME=DIR", help="prerun to link (repeatable)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    graph = KnowledgeGraph.load(args.graph)
    before = graph.summary() if hasattr(graph, "summary") else {}
    log.info("loaded %s: %d nodes / %d edges", args.graph,
             graph.g.number_of_nodes(), graph.g.number_of_edges())

    if args.merge_uses_from:
        n = merge_uses_edges(graph, Path(args.merge_uses_from))
        log.info("merged %d missing 'uses' edge(s) from %s", n,
                 args.merge_uses_from)

    ic_map: dict[str, float] = {}
    for spec in args.prerun:
        name, _, d = spec.partition("=")
        prerun_dir = Path(d or name)
        if not prerun_dir.exists():
            log.warning("SKIP %s: %s does not exist", name, prerun_dir)
            continue
        s = link_prerun(graph, name, prerun_dir, ic_map)
        log.info("%-28s factors=%-3d realized_by=%-3d (inherited=%d) "
                 "uses=%-3d no_mechanism=%d", name, s["factors"],
                 s["realized_by"], s["inherited"], s["uses"], s["no_mechanism"])

    refresh_mechanism_coverage(graph, ic_map)
    log.info("coverage refreshed over %d factor IC(s)", len(ic_map))
    log.info("graph now: %d nodes / %d edges (was %s)",
             graph.g.number_of_nodes(), graph.g.number_of_edges(), before)

    if args.dry_run:
        log.info("dry-run: NOT saving")
        return
    graph.save(args.out or args.graph)


if __name__ == "__main__":
    main()
