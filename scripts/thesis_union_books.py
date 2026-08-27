#!/usr/bin/env python
"""Results-table rows for the 101-alpha UNION books (user request 2026-08-21):

    zoo+arm4   101 formulaic alphas + L1H_terra_s0b  (arm 4, curated book)
    zoo+arm6   101 formulaic alphas + L4WF_terra_s0  (arm 6, PIT archive)
    zoo+all    101 alphas + every chapter arm 1-8 (the 4o-mini and GP arms
               are excluded as broken: look-ahead resp. metric gaming)

Same columns and definitions as scripts/thesis_ablation_master_table.py /
scripts/alpha_arms_table.py:

  * "prequential" column — a union book has no run of its own, so (exactly as
    for the zoo row of the master table) the PIT LightGBM race is reported as
    the analogue of the arms' prequential LightGBM refit record (flagged).
  * walk-forward (Lasso) — LassoCV in the point-in-time combiner race on the
    union book: seeding-only arms (children-per-deme 0) enter with their
    curated book from block 1 (`@full` + keep-fids, PIT-honest because every
    member is a generation-0 seed), evolution arms (L2WFP/L4WF/L5WF) with
    their replayed archive snapshot (`@snapshots`, book@gen g trades block
    g+1), the zoo from block 1.  Mean / s/sqrt(10) / hit over the 10 blocks.
  * per factor — median |mean block IC| and sign-flip share over the union's
    FINAL-book members with finite, non-zero fit IC (rows taken from the
    arms' own per-factor tables: per-factor ICs do not depend on the book).
  * independence — |B| = final union size, participation-ratio N_eff and
    mean |rho| on the fit window (<2021-07-20; strided 400 rows, flattened
    per-underlying, inf-safe nan_to_num — the wf_arm_factor_analysis
    convention) from the shared parquet signal store.
  * cost — N_trials and metered $ summed over the member arms (zoo: 0).

Outputs data/comparisons/thesis_union_books/{union_table.csv, union_table.tex,
members.json, keep_fids/*.json}; PIT races append to
data/comparisons/wf_arm_analysis_local/pit_combiners/<label>.jsonl
(resume-safe).  --skip-races reuses existing race summaries.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
ANA = REPO / "data/comparisons/wf_arm_analysis_local"
PIT = ANA / "pit_combiners"
TAB = REPO / "data/comparisons/thesis_ablation/tables"
STORE = REPO / "data/comparisons/wf_arm_analysis/signal_store"
OUT = REPO / "data/comparisons/thesis_union_books"
WF_START = pd.Timestamp("2021-07-20")

# chapter arm -> (prerun, availability, per-factor source)
#   per-factor source: ("all", <arm code in per_factor_all.csv>) or ("local", prerun)
ARMS = {
    "1": ("LDU8_terra_s0", "full", ("all", "1")),
    "2": ("LDP8_terra_s0", "full", ("all", "2")),
    "3": ("LDG_terra_s0b", "full", ("local", "LDG_terra_s0b")),
    "4": ("L1H_terra_s0b", "full", ("local", "L1H_terra_s0b")),
    "5": ("L2WFP_terra_s0", "snapshots", ("all", "5")),
    "6": ("L4WF_terra_s0", "snapshots", ("all", "6")),
    "7": ("L5WF_terra_s0", "snapshots", ("all", "7")),
    "8": ("L1HBD_terra_s0", "full", ("all", "8b")),
}
BOOKS = {
    "ZOO_plus_arm4": ("101 alphas + arm 4 (L1H)", ["4"]),
    "ZOO_plus_arm6": ("101 alphas + arm 6 (L4WF)", ["6"]),
    "ZOO_plus_all": ("101 alphas + arms 1-8", list(ARMS)),
}
METHODS = "equal,ic,ridge,lasso,lightgbm"


def zoo_book() -> dict[str, str]:
    pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
    return {m["factor_id"]: m["code"] for m in pb["members"]}


def final_book(prerun: str) -> dict[str, str]:
    db = json.loads((WS / prerun / "factors/factor_db.json").read_text())
    out = {}
    for r in db["factors"]:
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        out[r["id"]] = path.read_text()
    return out


def pool_fids(prerun: str) -> set[str]:
    """Every factor id an evolution arm ever kept (so keep-fids never trims
    a replayed snapshot)."""
    st_p = WS / prerun / "evolution/state.json"
    fids = set(final_book(prerun))
    if st_p.exists():
        st = json.loads(st_p.read_text())
        for key in ("kept_pool", "archive"):
            for eg in st.get(key, []):
                for prog in eg["genome"]["programs"]:
                    fids.add(prog["factor_id"])
        for grp in st.get("group_archives", []):
            for eg in grp:
                for prog in eg["genome"]["programs"]:
                    fids.add(prog["factor_id"])
    return fids


def run_race(label: str, arm_expr: str, keep_path: Path) -> None:
    cmd = [str(REPO / "venv/bin/python"), str(REPO / "scripts/wf_pit_combiner_study.py"),
           "--out-root", str(ANA), "--methods", METHODS,
           "--arm", arm_expr, "--label", label, "--keep-fids", str(keep_path)]
    print("[race]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=False)  # exits rc=1 after a good summary (known)
    if not (PIT / f"{label}_summary.csv").exists():
        raise SystemExit(f"race {label} produced no summary")


def pit(label: str, method: str) -> tuple[float, float, float, float]:
    df = pd.read_csv(PIT / f"{label}_summary.csv")
    r = df[df["method"] == method]
    if r.empty:
        return (math.nan,) * 4
    r = r.iloc[0]
    return (float(r["blockmean"]), float(r["blockstd"]) / math.sqrt(float(r["n_blocks"])),
            float(r["hit"]), float(r["mean_n_factors"]))


def per_factor_rows(arm_keys: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(ANA / "zoo/per_factor_blocks.csv").assign(src="zoo")]
    pfa = pd.read_csv(TAB / "per_factor_all.csv", dtype={"arm": str})
    for k in arm_keys:
        prerun, _, (kind, ref) = ARMS[k]
        book = set(final_book(prerun))
        if kind == "all":
            d = pfa[pfa["arm"] == ref]
        else:
            d = pd.read_csv(ANA / ref / "per_factor_blocks.csv")
        d = d[d["factor_id"].isin(book)]
        if len(d) < len(book):
            print(f"  [warn] arm {k}: per-factor rows {len(d)} < book {len(book)}")
        frames.append(d.assign(src=prerun))
    return pd.concat(frames, ignore_index=True)


def diversity(members: dict[str, str]) -> dict:
    from wf_common import signal_key
    ref_index = ref_cols = None
    vecs, missing = [], []
    for fid, code in members.items():
        p = STORE / f"{signal_key(fid, code)}.parquet"
        if not p.exists():
            missing.append(fid)
            continue
        sig = pd.read_parquet(p)
        if ref_index is None:
            fit_idx = sig.index[sig.index < WF_START]
            stride = max(1, len(fit_idx) // 400)
            ref_index = fit_idx[::stride]
            ref_cols = sig.columns
        v = sig.reindex(index=ref_index, columns=ref_cols).to_numpy(dtype=float).ravel()
        vecs.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    mat = np.column_stack(vecs)
    c = np.corrcoef(mat, rowvar=False)
    n = c.shape[0]
    off = c[np.triu_indices(n, 1)]
    off = off[np.isfinite(off)]
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(c)), 0, None)
    return {"n_factors": n, "n_missing_signal": len(missing), "missing": missing,
            "mean_abs_corr": float(np.mean(np.abs(off))),
            "max_abs_corr": float(np.max(np.abs(off))),
            "effective_n_participation_ratio": float(eig.sum() ** 2 / (eig ** 2).sum())}


def cost_and_trials(prerun: str) -> tuple[float, float]:
    man_p = WS / prerun / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    usage_p = WS / prerun / "evolution/llm_usage.json"
    usage = json.loads(usage_p.read_text()) if usage_p.exists() else man.get("llm_usage", {})
    cost = sum(v.get("cost_usd", 0.0) for v in usage.get("by_role", {}).values())
    return float(man.get("n_trials", math.nan)), cost


def latex(df: pd.DataFrame) -> str:
    out = []
    for _, r in df.iterrows():
        out.append(
            f"{r['arm']} & {r['preq_mean']:.4f} & {r['preq_se']:.4f} & {r['preq_hit']*10:.0f}/10 & "
            f"{r['lasso_book_mean']:.4f} & {r['lasso_book_se']:.4f} & {r['lasso_book_hit']*10:.0f}/10 & "
            f"{r['med_abs_ic']:.4f} & {r['flip_share']:.2f} & "
            f"{r['n_book']:.0f} & {r['n_eff']:.1f} & {r['mean_abs_corr']:.3f} & "
            f"{r['n_trials']:.0f} & {r['cost_usd']:.2f} \\\\")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="*", default=list(BOOKS))
    ap.add_argument("--skip-races", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "keep_fids").mkdir(exist_ok=True)

    rows, members_out = [], {}
    for label in args.books:
        name, arm_keys = BOOKS[label]
        print(f"=== {label}: {name}")
        members = dict(zoo_book())
        keep = set(members)
        arm_expr = ["zoo"]
        n_trials = cost = 0.0
        for k in arm_keys:
            prerun, mode, _ = ARMS[k]
            members.update(final_book(prerun))
            keep |= pool_fids(prerun) if mode == "snapshots" else set(final_book(prerun))
            arm_expr.append(f"{prerun}@{mode}")
            t, c = cost_and_trials(prerun)
            n_trials += t
            cost += c
        keep_path = OUT / "keep_fids" / f"{label}.json"
        keep_path.write_text(json.dumps(sorted(keep)))
        members_out[label] = {"name": name, "arms": [ARMS[k][0] for k in arm_keys],
                              "final_members": sorted(members), "n_final": len(members)}
        if not args.skip_races:
            run_race(label, "+".join(arm_expr), keep_path)

        lgb = pit(label, "lightgbm")
        las = pit(label, "lasso")
        pf = per_factor_rows(arm_keys)
        pf = pf[pf["factor_id"].isin(members)]
        ok = pf.dropna(subset=["ic_fit", "ic_wf_blockmean"])
        ok = ok[ok["ic_fit"] != 0]
        div = diversity(members)
        (OUT / f"{label}_diversity.json").write_text(json.dumps(div, indent=2))
        pf.to_csv(OUT / f"{label}_per_factor.csv", index=False)
        rows.append({
            "arm": name, "label": label,
            "preq_mean": lgb[0], "preq_se": lgb[1], "preq_hit": lgb[2],
            "preq_source": "PIT lightgbm (no own run)",
            "lasso_book_mean": las[0], "lasso_book_se": las[1], "lasso_book_hit": las[2],
            "lasso_book_n_avail": las[3],
            "med_abs_ic": float(ok["ic_wf_blockmean"].abs().median()),
            "flip_share": float((np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()),
            "flip_n": int(len(ok)),
            "n_book": len(members), "n_eff": div["effective_n_participation_ratio"],
            "mean_abs_corr": div["mean_abs_corr"], "n_trials": n_trials, "cost_usd": cost,
        })
        print(pd.Series(rows[-1]).to_string(), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "union_table.csv", index=False)
    (OUT / "union_table.tex").write_text(latex(df) + "\n")
    (OUT / "members.json").write_text(json.dumps(members_out, indent=1))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.drop(columns=["preq_source"]).to_string(index=False))
    print("\n% LaTeX rows\n" + latex(df))


if __name__ == "__main__":
    main()
