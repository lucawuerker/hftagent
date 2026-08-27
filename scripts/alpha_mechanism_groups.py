#!/usr/bin/env python
"""Build mechanism groups from the mechanisms the 101 formulaic alphas occupy.

The 101 Kakushadze/Yu formulaic alphas are NOT factor nodes in the knowledge
graph, so their mechanism coverage has to be established explicitly:

  1. every alpha's own documentation (name, category, description, formula
     docstring, inputs) is lexically pre-filtered against all mechanism nodes
     of the FROZEN ladder graph snapshot (TF-IDF over name+description) to a
     shared candidate pool;
  2. an LLM assigns each alpha the single best-matching mechanism from that
     pool (or ``none`` when no listed mechanism describes it);
  3. the occupied mechanisms are ranked by the highest absolute FIT-window IC
     among the alphas that occupy them (``ic_fit`` from the zoo book analysis —
     the fit window ends 2021-07-20, so the ranking carries no walk-forward
     look-ahead), and the top ``--n-groups`` become the run's mechanism groups;
  4. if the alphas occupy FEWER than ``--n-groups`` distinct mechanisms, the
     remainder is topped up from the ordinary graph-derived groups that the
     other ladder arms used (``mechanism_group_specs`` on the same snapshot),
     skipping any mechanism already selected.

Output is a group-spec JSON in exactly the shape ``resolve_mechanism_groups``
returns, consumable via ``run_factor_evolution.py --mechanism-groups-file``.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/knowledge/frozen/graph_wf_ladder_snapshot_2026-08-01.json"
PREBOOK = ROOT / "data/prebooks/formulaic_101.json"
ZOO_IC = ROOT / "data/comparisons/wf_arm_analysis_local/zoo/per_factor_blocks.csv"


# ── alpha documentation ──────────────────────────────────────────────────────
def load_alphas() -> list[dict]:
    members = json.loads(PREBOOK.read_text())["members"]
    out = []
    for m in members:
        code = m["code"]
        tree = ast.parse(code)
        doc = ast.get_docstring(tree) or ""
        rec = {"factor_id": m["factor_id"], "formula": doc.strip(),
               "name": "", "category": "", "description": "", "inputs": ""}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign) or not stmt.targets:
                    continue
                tgt = stmt.targets[0]
                if not isinstance(tgt, ast.Name) or tgt.id not in rec:
                    continue
                try:
                    val = ast.literal_eval(stmt.value)
                except Exception:
                    continue
                rec[tgt.id] = ", ".join(val) if isinstance(val, list) else str(val)
        out.append(rec)
    return out


def alpha_text(a: dict) -> str:
    return " ".join(x for x in (a["name"], a["category"].replace("_", " "),
                                a["description"], a["formula"], a["inputs"]) if x)


# ── graph mechanisms ─────────────────────────────────────────────────────────
def load_mechanisms(path: Path) -> list[dict]:
    g = json.loads(path.read_text())
    return [{"id": n["id"], "name": n.get("name", n["id"]),
             "description": n.get("description", ""),
             "community": n.get("community")}
            for n in g["nodes"] if n.get("type") == "mechanism"]


def mech_text(m: dict) -> str:
    return f"{m['name']}. {m['description']}"


# ── lexical prefilter ────────────────────────────────────────────────────────
def candidate_pool(alphas: list[dict], mechs: list[dict], per_alpha: int,
                   pool_max: int) -> list[dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [mech_text(m) for m in mechs] + [alpha_text(a) for a in alphas]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                          sublinear_tf=True, min_df=1)
    X = vec.fit_transform(corpus)
    M, A = X[:len(mechs)], X[len(mechs):]
    sim = (A @ M.T).toarray()                      # (n_alphas, n_mechs), L2-normed
    score: dict[int, float] = {}
    for row in sim:
        for j in row.argsort()[::-1][:per_alpha]:
            score[j] = max(score.get(j, 0.0), float(row[j]))
    ranked = sorted(score, key=lambda j: -score[j])[:pool_max]
    return [mechs[j] for j in ranked]


# ── LLM assignment ───────────────────────────────────────────────────────────
PROMPT = """You are mapping classic formulaic equity alphas onto the economic \
mechanisms of a research knowledge graph.

Below is a numbered CANDIDATE MECHANISM list, then a batch of ALPHAS. For every \
alpha, pick the ONE candidate mechanism that best describes the economic \
mechanism the alpha's formula is trying to exploit. Judge the mechanism, not the \
data fields: two alphas built from the same inputs can exploit different \
mechanisms, and the same mechanism can be reached by different formulas. If no \
candidate plausibly describes the alpha, answer "none".

CANDIDATE MECHANISMS
{mechanisms}

ALPHAS
{alphas}

Return ONLY a JSON array, one object per alpha, in the same order:
[{{"factor_id": "alpha_001", "mechanism_index": 12, "confidence": "high|medium|low", \
"why": "<one short sentence>"}}]
Use "mechanism_index": -1 for "none"."""


def assign(alphas: list[dict], pool: list[dict], model: str, provider: str,
           batch: int) -> dict[str, int]:
    import sys
    sys.path.insert(0, str(ROOT))
    from quant_fund_agent.llm import make_chat_llm

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    llm = make_chat_llm(model, provider, role="alpha_mechanism_map")
    mech_block = "\n".join(
        f"[{i}] {m['name']} — {m['description']}" for i, m in enumerate(pool))
    mapping: dict[str, int] = {}
    for start in range(0, len(alphas), batch):
        chunk = alphas[start:start + batch]
        alpha_block = "\n\n".join(
            f"{a['factor_id']} ({a['category']}): {a['name']}\n"
            f"  formula: {a['formula']}\n  doc: {a['description']}"
            for a in chunk)
        prompt = PROMPT.format(mechanisms=mech_block, alphas=alpha_block)
        text = ""
        for attempt in range(3):
            try:
                text = str(llm.invoke(prompt).content)
                rows = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
                for r in rows:
                    fid = str(r.get("factor_id", ""))
                    idx = int(r.get("mechanism_index", -1))
                    if fid and 0 <= idx < len(pool):
                        mapping[fid] = idx
                break
            except Exception as exc:                      # noqa: BLE001
                print(f"  batch {start//batch}: attempt {attempt+1} failed ({exc})")
        print(f"  batch {start//batch}: {len(mapping)} alphas mapped so far")
    return mapping


# ── group assembly ───────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groups", type=int, default=8)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--pool-max", type=int, default=300)
    ap.add_argument("--per-alpha", type=int, default=25)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--graph", default=str(SNAPSHOT))
    ap.add_argument("--out", default="data/knowledge/alpha_mechanism_groups.json")
    ap.add_argument("--map-out", default="data/knowledge/alpha_mechanism_map.csv")
    ap.add_argument("--ic-column", default="ic_fit")
    args = ap.parse_args()

    os.environ.setdefault("QF_GRAPH_PATH", args.graph)
    alphas = load_alphas()
    mechs = load_mechanisms(Path(args.graph))
    print(f"alphas={len(alphas)} mechanisms={len(mechs)}")

    pool = candidate_pool(alphas, mechs, args.per_alpha, args.pool_max)
    print(f"candidate pool: {len(pool)} mechanisms")

    mapping = assign(alphas, pool, args.model, args.provider, args.batch)
    print(f"assigned {len(mapping)}/{len(alphas)} alphas")

    ic = pd.read_csv(ZOO_IC).set_index("factor_id")
    rows = []
    for a in alphas:
        idx = mapping.get(a["factor_id"])
        m = pool[idx] if idx is not None else None
        rows.append({
            "factor_id": a["factor_id"], "category": a["category"],
            "mechanism_id": m["id"] if m else "",
            "mechanism": m["name"] if m else "",
            "community": m["community"] if m else "",
            "ic_fit": float(ic.at[a["factor_id"], "ic_fit"])
            if a["factor_id"] in ic.index else float("nan"),
            "ic_wf_blockmean": float(ic.at[a["factor_id"], "ic_wf_blockmean"])
            if a["factor_id"] in ic.index else float("nan"),
        })
    dfm = pd.DataFrame(rows)
    Path(args.map_out).parent.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(args.map_out, index=False)

    occupied = dfm[dfm["mechanism_id"] != ""].copy()
    occupied["abs_ic"] = occupied[args.ic_column].abs()
    rank = (occupied.groupby(["mechanism_id", "mechanism", "community"])
            .agg(max_abs_ic=("abs_ic", "max"), n_alphas=("factor_id", "size"),
                 alphas=("factor_id", lambda s: ",".join(sorted(s))))
            .reset_index().sort_values("max_abs_ic", ascending=False))
    print(f"\ndistinct mechanisms occupied: {len(rank)}")
    print(rank.head(args.n_groups).to_string(index=False))

    by_id = {m["id"]: m for m in mechs}
    chosen = rank.head(args.n_groups)
    specs: list[dict] = []
    for _, r in chosen.iterrows():
        m = by_id[r["mechanism_id"]]
        # sibling mechanisms other alphas occupy inside the same community
        sib = [by_id[s]["name"] for s in occupied["mechanism_id"].unique()
               if s != m["id"] and by_id[s].get("community") == m.get("community")]
        targets = [m["name"], *sib[:3]]
        focus = (f"{m['name']}: {m['description']}"
                 + "\ntarget mechanisms: " + ", ".join(targets))
        specs.append({
            "mechanism_group_id": len(specs),
            "community_id": m.get("community"),
            "focus": focus,
            "mechanisms": targets,
            "source": "formulaic_101",
            "seed_mechanism_id": m["id"],
            "seed_alphas": r["alphas"],
            "max_abs_ic": float(r["max_abs_ic"]),
        })

    if len(specs) < args.n_groups:
        import sys
        sys.path.insert(0, str(ROOT))
        from quant_fund_agent.knowledge.graph_query import mechanism_group_specs
        from quant_fund_agent.knowledge.graph_store import KnowledgeGraph
        graph = KnowledgeGraph.load(Path(args.graph))
        taken = {s["seed_mechanism_id"] for s in specs}
        for spec in mechanism_group_specs(graph, args.n_groups * 3):
            if len(specs) >= args.n_groups:
                break
            if not spec.get("focus") or spec.get("community_id") in {
                    s["community_id"] for s in specs}:
                continue
            spec = dict(spec, mechanism_group_id=len(specs), source="graph_topup")
            specs.append(spec)
        print(f"topped up to {len(specs)} groups from the ordinary graph ranking")

    Path(args.out).write_text(json.dumps(specs, indent=2))
    print(f"\nwrote {len(specs)} groups -> {args.out}")
    print(f"wrote alpha->mechanism map -> {args.map_out}")


if __name__ == "__main__":
    main()
