# Frozen knowledge-graph snapshots (thesis artifacts)

These files are immutable references for the Master's thesis. Do **not**
regenerate or overwrite them — `data/knowledge/graph.json` is the live,
evolving graph; these are the states the experiments actually saw.

## `graph_wf_ladder_snapshot_2026-08-01.json`
The exact graph deployed to the lagias server on 2026-08-01 for the
walk-forward Terra ladder (`matrix/terra_wf_ladder.yaml`). Every WF arm
(L1WF–L7WF, plus the ablation-QA arms) ran with `--graph-readonly` and
resolved its mechanism groups from **this identical snapshot** — it is the
graph to cite for the ladder's group definitions and retrieval grounding.
Contents: 1,723 papers, 2,296 mechanisms, 81 anomalies, 54 fields,
36 factor nodes (link-backs from the pre-ladder local runs), 8,363 edges.

## `graph_local_pre_linkback_2026-08-13.json`
The local graph as of 2026-08-13, immediately **before** the post-hoc factor
link-back (`scripts/link_factors_into_graph.py`) that added the completed
runs' published books (859 factor nodes afterwards). Differs from the ladder
snapshot only in the factor/empirical layer: 8 extra factor nodes from local
runs after 2026-08-01, and it is missing the ladder snapshot's 71
`factor→field` `uses` edges (an in-run `refresh_field_usage` call rebuilds
that edge layer from scratch and had clobbered them — repaired in the live
graph by the link-back's `--merge-uses-from` pass). The semantic layer
(papers/mechanisms/anomalies) is byte-identical in both snapshots.

## Provenance of the live graph after 2026-08-13
`data/knowledge/graph.json` = ladder snapshot ∪ local factor nodes ∪ the
published factor books of all completed arms (L1/L2/L4 locals, L4WF–L7WF,
L1WF oneshot s0/s1, L2WFB/P, L4IC, GLD HF), with `realized_by` edges where a
mechanism stamp existed or could be inherited through the genome parent
chain (edge attr `provenance: seed|inherited`), `uses` edges from declared
`required_inputs`, and refreshed per-mechanism coverage stats.
