#!/usr/bin/env python
"""Per-mechanism comparison: the LLM factors bred inside the mechanisms the 101
formulaic alphas occupy, versus the alphas that occupy those same mechanisms.

Both sides are scored on the identical panel and the identical 10 prequential
walk-forward blocks (wf_arm_factor_analysis output), so the only difference is
who wrote the factor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANA = ROOT / "data/comparisons/wf_arm_analysis_local"
WS = ROOT / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
ARM = "L1HA_terra_s0b"
OUT = ROOT / "data/comparisons/alpha_arms"


def stats(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    if ok.empty:
        return {"n": 0, "med_abs_ic": np.nan, "max_abs_ic": np.nan,
                "flip": np.nan, "hit": np.nan}
    return {
        "n": len(ok),
        "med_abs_ic": float(ok["ic_wf_blockmean"].abs().median()),
        "max_abs_ic": float(ok["ic_wf_blockmean"].abs().max()),
        "flip": float((np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()),
        "hit": float(ok["wf_hit_rate"].median()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = json.loads((WS / ARM / "evolution/mechanism_groups.json").read_text())
    st = json.loads((WS / ARM / "evolution/state.json").read_text())

    # factor_id -> group, for the curated archive and for the whole kept pool
    arch, pool = {}, {}
    for gi, grp in enumerate(st["group_archives"]):
        for e in grp:
            for p in e["genome"]["programs"]:
                arch[p["factor_id"]] = gi
    for e in st.get("kept_pool", []):
        gi = e["genome"].get("mechanism_group_id")
        for p in e["genome"]["programs"]:
            pool.setdefault(p["factor_id"], gi)

    llm = pd.read_csv(ANA / ARM / "per_factor_blocks.csv")
    zoo = pd.read_csv(ANA / "zoo/per_factor_blocks.csv")
    amap = pd.read_csv(ROOT / "data/knowledge/alpha_mechanism_map.csv")

    rows = []
    for gi, g in enumerate(groups):
        mech = g["mechanisms"][0]
        alphas = amap[amap["mechanism"] == mech]["factor_id"].tolist()
        a = stats(zoo[zoo["factor_id"].isin(alphas)])
        l_arch = stats(llm[llm["factor_id"].isin(
            [f for f, x in arch.items() if x == gi])])
        rows.append({
            "group": gi, "mechanism": mech,
            "alphas_n": a["n"], "alphas_med_abs_ic": a["med_abs_ic"],
            "alphas_max_abs_ic": a["max_abs_ic"], "alphas_flip": a["flip"],
            "llm_book_n": l_arch["n"], "llm_book_med_abs_ic": l_arch["med_abs_ic"],
            "llm_book_max_abs_ic": l_arch["max_abs_ic"], "llm_book_flip": l_arch["flip"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_mechanism_alpha_vs_llm.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("=== per mechanism: the 101 alphas vs the LLM factors bred there ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # book-level totals
    print("\n=== book level (same 10 walk-forward blocks) ===")
    mapped = amap[amap["mechanism"] != ""]["factor_id"].tolist()
    top8 = [g["mechanisms"][0] for g in groups]
    top8_alphas = amap[amap["mechanism"].isin(top8)]["factor_id"].tolist()
    for label, d in (("all 101 alphas", zoo),
                     ("alphas in the 8 chosen mechanisms",
                      zoo[zoo["factor_id"].isin(top8_alphas)]),
                     (f"{ARM} curated book", llm[llm["factor_id"].isin(arch)]),
                     ):
        s = stats(d)
        print(f"  {label:38s} n={s['n']:3d}  median |IC| {s['med_abs_ic']:.4f}  "
              f"max |IC| {s['max_abs_ic']:.4f}  flip {s['flip']:.2f}  "
              f"median hit {s['hit']:.2f}")


if __name__ == "__main__":
    main()
