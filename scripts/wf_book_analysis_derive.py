"""Derive cross-arm book-analysis tables for the WF ladder (L2/L4/L5/L7).

Inputs (pulled from lagias): data/comparisons/wf_book_analysis/raw/<arm>/
  analysis/{per_factor_blocks.csv,combined_static.csv,diversity.json,prequential_record.csv}
  evolution/{state.json,lineage.jsonl,prequential.jsonl,gen_quality.jsonl}
  pit/{<arm>.jsonl,artifacts/g*/weights_lasso.json}

Outputs: data/comparisons/wf_book_analysis/derived/
  pareto_trajectories.csv   one row per (arm, genome, generation) with the 4 axes
  archive_members.csv       final archive: factor, group, admission gen, final axes
  lasso_selection.csv       per (arm, block): n selected, alpha, ic
  lasso_factor_freq.csv     per (arm, factor): selection count /10, phase counts
  combined_static_all.csv   concatenated static combiner rows
  pit_summary_all.csv       concatenated PIT combiner summaries
  diversity_all.csv         per-arm diversity incl. effective N
  per_factor_summary.csv    per-factor fit/WF block IC summary, all arms
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/comparisons/wf_book_analysis"
RAW = BASE / "raw"
OUT = BASE / "derived"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["L2WF_terra_s0", "L4WF_terra_s0", "L5WF_terra_s0", "L7WF_terra_s0"]
# ablation-QA arms (local workspaces; analysed into wf_arm_analysis_local)
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
LOCAL_ANALYSIS = REPO / "data/comparisons/wf_arm_analysis_local"
ABLATION_ARMS = {"L1H_terra_s0": "evolution", "L1HB_terra_s0": "evolution",
                 "L4D_terra_s0": "evolution", "L0WF_gp_s0": "gp",
                 "L4IC_terra_s0": "evolution", "L2WFB_terra_s0": "evolution",
                 "L2WFP_terra_s0": "evolution"}
AXES = ["marginal_value", "independence", "parsimony", "structural_novelty"]
# 10 prequential blocks g11..g20 -> three phases
PHASES = {"P1": [11, 12, 13], "P2": [14, 15, 16, 17], "P3": [18, 19, 20]}


def load_lineage(arm: str) -> list[dict]:
    return [json.loads(l) for l in (RAW / arm / "evolution/lineage.jsonl").open()]


def trajectories(arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    st = json.load((RAW / arm / "evolution/state.json").open())
    rows = load_lineage(arm)
    members = {}
    for e in st["archive"]:
        g, f = e["genome"], e["fitness"]
        fids = [p.get("factor_id") for p in g.get("programs", [])]
        members[g["genome_id"]] = {
            "arm": arm,
            "genome_id": g["genome_id"],
            "factor_id": fids[0] if fids else None,
            "group": g.get("mechanism_group_id"),
            "admit_generation": g.get("generation"),
            "operator": g.get("operator"),
            **{f"final_{a}": (f.get("objective") or {}).get(a) for a in AXES},
        }
    traj = []
    for r in rows:
        gid = r.get("genome_id")
        if gid not in members:
            continue
        if r.get("event") is None:
            obj = r.get("objective") or {}
            traj.append({"arm": arm, "genome_id": gid, "generation": r["generation"],
                         "event": "admit", **{a: obj.get(a) for a in AXES}})
        elif r.get("event") == "rescore":
            obj = r.get("objective_after") or {}
            traj.append({"arm": arm, "genome_id": gid, "generation": r["generation"],
                         "event": "rescore", **{a: obj.get(a) for a in AXES}})
    tdf = pd.DataFrame(traj).sort_values(["genome_id", "generation"])
    mdf = pd.DataFrame(members.values())
    if not tdf.empty:
        tdf["factor_id"] = tdf["genome_id"].map({k: v["factor_id"] for k, v in members.items()})
    return tdf, mdf


def lasso_tables(arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_block, freq = [], {}
    pit_rows = [json.loads(l) for l in (RAW / arm / "pit" / f"{arm}.jsonl").open()]
    lasso_rows = {r["block_gen"]: r for r in pit_rows if r.get("method") == "lasso"}
    for gdir in sorted((RAW / arm / "pit/artifacts").glob("g*")):
        blk = int(gdir.name[1:])
        if not (gdir / "weights_lasso.json").exists():
            continue
        w = json.load((gdir / "weights_lasso.json").open())
        fids, ws = w["fids"], np.asarray(w["weights"], dtype=float)
        sel = [f for f, x in zip(fids, ws) if x != 0.0]
        pr = lasso_rows.get(blk, {})
        per_block.append({"arm": arm, "block": blk, "n_avail": len(fids),
                          "n_selected": len(sel),
                          "alpha": (pr.get("diag") or {}).get("alpha"),
                          "ic_oos": pr.get("ic"), "ic_fit": pr.get("ic_fit")})
        for f in fids:
            freq.setdefault(f, {"arm": arm, "factor_id": f, "n_blocks_avail": 0,
                                "n_selected": 0, **{p: 0 for p in PHASES}})
        for f in fids:
            freq[f]["n_blocks_avail"] += 1
        for f in sel:
            freq[f]["n_selected"] += 1
            for p, blks in PHASES.items():
                if blk in blks:
                    freq[f][p] += 1
    return pd.DataFrame(per_block), pd.DataFrame(freq.values())


def main() -> None:
    all_traj, all_members, all_blocks, all_freq = [], [], [], []
    all_static, all_pit, all_div, all_pf = [], [], [], []
    for arm in ARMS:
        tdf, mdf = trajectories(arm)
        all_traj.append(tdf)
        all_members.append(mdf)
        bl, fr = lasso_tables(arm)
        all_blocks.append(bl)
        all_freq.append(fr)
        all_static.append(pd.read_csv(RAW / arm / "analysis/combined_static.csv"))
        s = pd.read_csv(RAW / arm / "pit" / f"{arm}_summary.csv")
        s.insert(0, "arm", arm)
        all_pit.append(s)
        d = json.load((RAW / arm / "analysis/diversity.json").open())
        d.pop("failed", None)
        all_div.append({"arm": arm, **d})
        pf = pd.read_csv(RAW / arm / "analysis/per_factor_blocks.csv")
        pf.insert(0, "arm", arm)
        all_pf.append(pf)

    pd.concat(all_traj).to_csv(OUT / "pareto_trajectories.csv", index=False)
    pd.concat(all_members).to_csv(OUT / "archive_members.csv", index=False)
    pd.concat(all_blocks).to_csv(OUT / "lasso_selection.csv", index=False)
    pd.concat(all_freq).to_csv(OUT / "lasso_factor_freq.csv", index=False)
    pd.concat(all_static).to_csv(OUT / "combined_static_all.csv", index=False)
    pd.concat(all_pit).to_csv(OUT / "pit_summary_all.csv", index=False)
    pd.DataFrame(all_div).to_csv(OUT / "diversity_all.csv", index=False)
    pd.concat(all_pf).to_csv(OUT / "per_factor_summary.csv", index=False)

    ladder_summary()

    for arm in ARMS:
        m = pd.concat(all_members).query("arm == @arm")
        b = pd.concat(all_blocks).query("arm == @arm")
        print(f"{arm}: archive {len(m)}, lasso mean selected "
              f"{b['n_selected'].mean():.1f}/{b['n_avail'].mean():.0f}")


def _prequential_blocks(path: Path) -> list[float]:
    rows = [json.loads(l) for l in path.open()]
    return [r["combined_oos_ic"] for r in rows
            if r.get("generation", 0) >= 11 and r.get("combined_oos_ic") is not None]


def ladder_summary() -> None:
    """One row per arm across the whole ablation ladder: honest prequential
    record, PIT-race best (where raced), book size, trials, LLM cost."""
    # constants for numbers whose source files are off-box (server SSH down
    # 2026-08-10) — from docs/research-evolution/ABLATION_QA.md final table
    KNOWN = {
        "L1WF_oneshot_terra_s0": dict(pit_best=0.078, pit_best_method="lasso",
                                      n_book=107, trials=107, cost_usd=20.86),
        "L4WF_terra_s0": dict(cost_usd=250.0, trials=800),
        "L2WF_terra_s0": dict(cost_usd=None, trials=None),
    }
    rows = []
    arms = (["L0WF_gp_s0", "L1WF_oneshot_terra_s0", "L1H_terra_s0",
             "L1HB_terra_s0", "L2WF_terra_s0", "L2WFB_terra_s0",
             "L2WFP_terra_s0",
             "L4D_terra_s0", "L4IC_terra_s0", "L4WF_terra_s0",
             "L5WF_terra_s0", "L7WF_terra_s0"])
    for arm in arms:
        sub = ABLATION_ARMS.get(arm, "evolution")
        row: dict = {"arm": arm}
        preq_path = (RAW / arm / "evolution/prequential.jsonl")
        if not preq_path.exists():
            preq_path = WS / arm / sub / "prequential.jsonl"
        if preq_path.exists():
            ics = _prequential_blocks(preq_path)
            if ics:
                row["preq_mean_ic"] = float(np.mean(ics))
                row["preq_se"] = float(np.std(ics, ddof=1) / np.sqrt(len(ics)))
                row["preq_hit"] = float(np.mean([x > 0 for x in ics]))
                row["preq_blocks"] = len(ics)
        st_path = RAW / arm / "evolution/state.json"
        if not st_path.exists():
            st_path = WS / arm / sub / "state.json"
        if st_path.exists():
            st = json.load(st_path.open())
            row["n_book"] = len(st.get("archive", []))
            row["trials"] = st.get("n_trials")
        u_path = WS / arm / "evolution/llm_usage.json"
        if u_path.exists():
            row["cost_usd"] = json.load(u_path.open())["total"]["cost_usd"]
        # PIT race best over available summaries (server pull + local runs)
        best = None
        for cand in [RAW / arm / "pit" / f"{arm}_summary.csv",
                     LOCAL_ANALYSIS / "pit_combiners" / f"{arm}_summary.csv"]:
            if cand.exists():
                s = pd.read_csv(cand)
                idx = s["blockmean"].idxmax()
                best = (float(s.loc[idx, "blockmean"]), str(s.loc[idx, "method"]))
        if best:
            row["pit_best"], row["pit_best_method"] = best
        row.update({k: v for k, v in KNOWN.get(arm, {}).items()
                    if row.get(k) is None or k not in row})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "ladder_summary.csv", index=False)


if __name__ == "__main__":
    main()
