"""Build (or extend) the hybrid knowledge graph from the paper library.

One LLM extraction call per paper → mechanisms / anomalies / field needs, then
community detection + summaries.  Idempotent: papers already in the graph are
skipped, so re-running after adding papers only pays for the new ones.

The mechanism list is printed at the end for the human spot-check the design
recommends (extraction noise / near-duplicate mechanisms are THE known risk of
GraphRAG builds — skim the list, then merge outliers by editing the graph JSON
or re-running with a better model).

Usage::

    ./venv/bin/python scripts/build_knowledge_graph.py --max-papers 25   # trial
    ./venv/bin/python scripts/build_knowledge_graph.py                   # full
    ./venv/bin/python scripts/build_knowledge_graph.py --model gpt-4o \
        --llm-summaries   # better extraction + prose community summaries
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_knowledge_graph")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=None,
                   help="Extraction LLM (default: the research model).")
    p.add_argument("--llm-provider", default=None)
    p.add_argument("--max-papers", type=int, default=None,
                   help="Cap the number of NEW papers extracted this run.")
    p.add_argument("--graph-path", default=None,
                   help="Graph JSON path (default data/knowledge/graph.json).")
    p.add_argument("--llm-summaries", action="store_true",
                   help="Write community summaries with the LLM (default: "
                        "deterministic member lists).")
    args = p.parse_args()

    if args.model:
        os.environ["FACTOR_RESEARCH_LLM_MODEL"] = args.model
    if args.llm_provider:
        os.environ["FACTOR_RESEARCH_LLM_PROVIDER"] = args.llm_provider

    from quant_fund_agent.knowledge.embed_store import load_corpus
    from quant_fund_agent.knowledge.graph_build import (
        build_graph,
        detect_and_summarize_communities,
    )
    from quant_fund_agent.knowledge.graph_store import (
        DEFAULT_GRAPH_PATH,
        KnowledgeGraph,
    )
    from quant_fund_agent.llm import make_chat_llm

    path = Path(args.graph_path) if args.graph_path else DEFAULT_GRAPH_PATH
    graph = KnowledgeGraph.load(path) if path.exists() else KnowledgeGraph()
    log.info("starting from: %s", graph.summary())

    docs = load_corpus()
    log.info("corpus: %d paper(s)", len(docs))
    llm = make_chat_llm(temperature=0.1, timeout=120, max_retries=4)

    # Checkpoint into the real graph path every 25 papers: a crash or an LLM
    # budget-ceiling stop keeps all paid-for extractions, and re-running
    # resumes (ingested papers are skipped).
    graph = build_graph(llm, docs, graph=graph, max_papers=args.max_papers,
                        checkpoint_path=path)
    summaries = detect_and_summarize_communities(
        graph, llm=llm if args.llm_summaries else None)
    graph.save(path)

    print("\n" + "=" * 80)
    print(f"Knowledge graph: {path}")
    print(f"  {graph.summary()}")
    print(f"  {len(summaries)} communities")
    print("\nMechanisms (spot-check for near-duplicates!):")
    for m in sorted(graph.nodes_of_type("mechanism")):
        d = graph.g.nodes[m]
        n_papers = len(graph.papers_for_mechanism(m))
        print(f"  - {d.get('name', m)}  [{n_papers} paper(s), "
              f"community {d.get('community', '?')}]")
    print("=" * 80)


if __name__ == "__main__":
    main()
