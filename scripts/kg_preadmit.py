"""One-time pre-admission for the KG breadth campaign: link every previous
research book into the LIVE knowledge graph (runs already linked are no-ops —
existing factor nodes are refreshed, not duplicated). Also writes the
campaign's cumulative_book.json seed (run=0 entries) so run-1 dedup sees the
whole history.

Books: all researcher preruns with a factor_db (mechanism-tagged books link
Paper->Mechanism->Factor->Field; untagged ones — LDU8/LDP8 neutral-group and
oneshot books — contribute factor/field nodes only). The 101-alpha zoo is
EXCLUDED (formulaic, no mechanisms, not LLM research output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_preadmit")

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
CAMP = REPO / "data/kg_campaign"
BOOKS = ["L1WF_oneshot_terra_s0", "L1H_terra_s0", "L1HB_terra_s0",
         "L1HBD_terra_s0", "L1HB4OMP_terra_s0", "L2WF_terra_s0",
         "L2WFP_terra_s0", "L4WF_terra_s0", "L4D_terra_s0", "L4IC_terra_s0",
         "L5WF_terra_s0", "L6WF_terra_s0", "L7WF_terra_s0",
         "LDG_terra_s0", "LDP_terra_s0", "LDU_terra_s0",
         "LDP8_terra_s0", "LDU8_terra_s0"]


def main() -> None:
    from quant_fund_agent.agents.factor_research.evolution.genome import (
        FactorProgram,
    )
    from quant_fund_agent.agents.factor_research.evolution.loop import (
        link_programs_into_graph,
    )
    from quant_fund_agent.knowledge.graph_store import KnowledgeGraph

    graph = KnowledgeGraph.load()
    CAMP.mkdir(parents=True, exist_ok=True)
    cum_path = CAMP / "cumulative_book.json"
    cum = (json.loads(cum_path.read_text()) if cum_path.exists() else [])
    have = {e["factor_id"] for e in cum}

    total_linked = 0
    for arm in BOOKS:
        db_path = WS / arm / "factors/factor_db.json"
        if not db_path.exists():
            log.warning("[%s] no factor_db — skipped", arm)
            continue
        db = json.loads(db_path.read_text())
        programs, mech_by_fid = [], {}
        for r in db["factors"]:
            p = Path(r["code_path"])
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                p = REPO / "quant_fund_agent/factors/researcher" / p.name
            if not p.exists():
                log.warning("[%s] %s: code missing", arm, r["id"])
                continue
            code = p.read_text()
            programs.append(FactorProgram(
                factor_id=r["id"], code=code, name=r.get("name", r["id"]),
                trading_idea=str(r.get("trading_idea") or ""),
                expected_sign=int(r.get("expected_sign") or 0) or 1,
                prediction_horizon=int(r.get("prediction_horizon") or 6)))
            mech = (r.get("metadata") or {}).get("mechanism")
            if mech:
                mech_by_fid[r["id"]] = mech
            if r["id"] not in have:
                cum.append({"factor_id": r["id"], "run": 0,
                            "source_book": arm, "code_path": str(p)})
                have.add(r["id"])
        if programs:
            link_programs_into_graph(graph, programs, mech_by_fid,
                                     readonly=False)
            total_linked += len(programs)
            log.info("[%s] linked %d (mech-tagged %d)", arm, len(programs),
                     len(mech_by_fid))
    graph.save()
    cum_path.write_text(json.dumps(cum, indent=1))
    log.info("DONE: %d programs linked; cumulative seed book %d factors",
             total_linked, len(cum))


if __name__ == "__main__":
    main()
