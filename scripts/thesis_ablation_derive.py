"""Assemble every table the thesis ablation chapter (Experiments & Results)
reports, for the nine ladder arms + the model-quality arm + the KG campaign.

Reported per arm (chapter definitions):
  * prequential block ICs (gens 11-20), block mean, SE = s/sqrt(10), hit rate H
  * WF IC under the sparse linear combiner (PIT LassoCV race), book and pool
  * lasso selection: mean #nonzero, share of available members
  * flip share Phi over published-book members (sign(ic_fit) vs sign(mean WF))
  * book independence: mean |pairwise rho|, N_eff (participation ratio)
  * N_trials and metered cost in USD

Outputs under data/comparisons/thesis_ablation/tables/:
  arms_meta.csv, prequential_blocks.csv, ladder_summary.csv,
  pit_methods.csv, costs_by_role.csv, per_factor_all.csv,
  kg_campaign.csv, model_quality_blocks.csv
plus LaTeX snippets under .../tables/tex/.

Safe to re-run: recomputes everything from the source artifacts each time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
LOCAL = REPO / "data/comparisons/wf_arm_analysis_local"
SERVER = REPO / "data/comparisons/wf_arm_analysis_server"
RAW = REPO / "data/comparisons/wf_book_analysis/raw"
OUT = REPO / "data/comparisons/thesis_ablation/tables"
TEX = OUT / "tex"

# ── arm registry ───────────────────────────────────────────────────────────
# thesis arm number -> (run label, evolution dir, analysis dir, pit files)
ARMS: dict[str, dict] = {
    "1": dict(run="LDU8_terra_s0", name="LDU8",
              grounding="none", evolution=False, review=False,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "2": dict(run="LDP8_terra_s0", name="LDP8",
              grounding="24 random papers", evolution=False, review=False,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "3": dict(run="LDG_terra_s0", name="LDG",
              grounding="graph briefs", evolution=False, review=False,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "4": dict(run="L1H_terra_s0", name="L1H",
              grounding="graph + papers", evolution=False, review=False,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "5": dict(run="L2WFP_terra_s0", name="L2WFP",
              grounding="none", evolution=True, review=False,
              groups="1x4", seeds=96, llm="GPT-5.6 Terra"),
    "6": dict(run="L4WF_terra_s0", name="L4WF",
              grounding="graph + papers", evolution=True, review=False,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "7": dict(run="L5WF_terra_s0", name="L5WF",
              grounding="graph + papers", evolution=True, review=True,
              groups="8x3", seeds=96, llm="GPT-5.6 Terra"),
    "8a": dict(run="L1HB_terra_s0", name="L1HB",
               grounding="graph + papers", evolution=False, review=False,
               groups="8x3", seeds=192, llm="GPT-5.6 Terra"),
    "8b": dict(run="L1HBD_terra_s0", name="L1HBD",
               grounding="graph + papers", evolution=False, review=True,
               groups="8x3", seeds=192, llm="GPT-5.6 Terra"),
    "9": dict(run="L0WF_gp_s0", name="L0WF",
              grounding="none (random trees)", evolution=True, review=False,
              groups="1x3", seeds=96, llm="none (GP)"),
    "MQ": dict(run="L1HB_4omini_s0", name="L1HB-4o-mini",
               grounding="graph + papers", evolution=False, review=False,
               groups="8x3", seeds=192, llm="GPT-4o-mini"),
}

# per-run source resolution
def _evo_candidates(run: str) -> list[Path]:
    return [WS / run / "evolution", RAW / run / "evolution", WS / run / "gp"]


def evo_dir(run: str) -> Path | None:
    for cand in _evo_candidates(run):
        if (cand / "prequential.jsonl").exists():
            return cand
    return None


def evo_file(run: str, name: str) -> Path | None:
    for cand in _evo_candidates(run):
        if (cand / name).exists():
            return cand / name
    return None


def analysis_dir(run: str) -> Path | None:
    for cand in (LOCAL / run, SERVER / run, RAW / run / "analysis"):
        if (cand / "per_factor_blocks.csv").exists():
            return cand
    return None


def pit_paths(run: str) -> dict[str, Path]:
    """label kind -> jsonl.  'book' = curated/published book (CUR pseudo-
    prerun or snapshot replay for evolution arms), 'pool' = kept pool."""
    base = run.replace("_terra_s0", "").replace("_gp_s0", "")
    out: dict[str, Path] = {}
    if "4omini" in run:
        book_cands = [LOCAL / "pit_combiners/L1HB4OMINICUR_s0.jsonl"]
    else:
        book_cands = [LOCAL / f"pit_combiners/{base}CUR_terra_s0.jsonl"]
    cands = {
        "pool": [LOCAL / f"pit_combiners/{run}.jsonl",
                 SERVER / f"pit_combiners/{run}.jsonl"],
        "book": book_cands,
    }
    # evolution arms: the snapshot race IS the curated-book race
    if run in ("L2WFP_terra_s0", "L4WF_terra_s0", "L5WF_terra_s0",
               "L0WF_gp_s0"):
        cands["book"] = [RAW / run / f"pit/{run}.jsonl",
                         LOCAL / f"pit_combiners/{run}.jsonl"]
        cands["pool"] = []
    for kind, paths in cands.items():
        for p in paths:
            if p.exists():
                out[kind] = p
                break
    return out


def read_prequential(run: str) -> pd.DataFrame:
    d = evo_dir(run)
    if d is None:
        return pd.DataFrame()
    rows = []
    for line in (d / "prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            rows.append(dict(run=run, block=r["generation"] - 10,
                             generation=r["generation"],
                             start=r["start"][:10], end=r["end"][:10],
                             ic=r["combined_oos_ic"], n_obs=r.get("n_obs")))
    return pd.DataFrame(rows)


def read_pit(path: Path, method: str = "lasso") -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("method") != method or r.get("block_gen", 0) < 11:
            continue
        diag = r.get("diag") or {}
        rows.append(dict(block=r["block_gen"] - 10, ic=r.get("ic"),
                         n_avail=r.get("n_factors_avail"),
                         n_selected=diag.get("n_nonzero_coef",
                                             diag.get("n_nonzero",
                                                      diag.get("n_selected")))))
    return pd.DataFrame(rows)


def read_pit_all_methods(path: Path, label: str, kind: str) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("block_gen", 0) < 11:
            continue
        rows.append(dict(run=label, book=kind, method=r["method"],
                         block=r["block_gen"] - 10, ic=r.get("ic"),
                         n_avail=r.get("n_factors_avail")))
    return pd.DataFrame(rows)


def read_state(run: str) -> dict:
    p = evo_file(run, "state.json")
    if p is None:
        return {}
    st = json.loads(p.read_text())
    return dict(n_trials=st.get("n_trials"),
                n_archive=len(st.get("archive", [])),
                n_pool=len(st.get("kept_pool", [])))


def read_cost(run: str) -> tuple[float | None, dict]:
    p = evo_file(run, "llm_usage.json")
    if p is None:
        if "_gp_" in run:
            return 0.0, {}
        return None, {}
    u = json.loads(p.read_text())
    roles = {k: v.get("cost_usd", 0.0) for k, v in u.get("by_role", {}).items()}
    total = u.get("total_cost_usd") or sum(roles.values())
    return total, roles


def flip_share(run: str) -> tuple[float | None, int | None]:
    d = analysis_dir(run)
    if d is None:
        return None, None
    df = pd.read_csv(d / "per_factor_blocks.csv")
    ok = df.dropna(subset=["ic_fit", "ic_wf_blockmean"])
    ok = ok[ok["ic_fit"] != 0]
    if not len(ok):
        return None, None
    flips = (np.sign(ok["ic_fit"]) != np.sign(ok["ic_wf_blockmean"])).mean()
    return float(flips), int(len(ok))


def diversity(run: str) -> dict:
    d = analysis_dir(run)
    if d is None or not (d / "diversity.json").exists():
        return {}
    return json.loads((d / "diversity.json").read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)

    # 1. arm registry --------------------------------------------------------
    meta = pd.DataFrame([dict(arm=k, **v) for k, v in ARMS.items()])
    meta.to_csv(OUT / "arms_meta.csv", index=False)

    # 2. prequential blocks --------------------------------------------------
    preq = pd.concat([read_prequential(v["run"]) for v in ARMS.values()],
                     ignore_index=True)
    preq.to_csv(OUT / "prequential_blocks.csv", index=False)

    # 3. PIT races (all methods, long form) ---------------------------------
    pit_frames = []
    for v in ARMS.values():
        for kind, path in pit_paths(v["run"]).items():
            pit_frames.append(read_pit_all_methods(path, v["run"], kind))
    pit_all = (pd.concat(pit_frames, ignore_index=True)
               if pit_frames else pd.DataFrame())
    pit_all.to_csv(OUT / "pit_methods.csv", index=False)

    # 4. ladder summary ------------------------------------------------------
    rows = []
    for k, v in ARMS.items():
        run = v["run"]
        p = preq[preq["run"] == run]
        st = read_state(run)
        cost, roles = read_cost(run)
        div = diversity(run)
        phi, n_phi = flip_share(run)
        row = dict(arm=k, name=v["name"], run=run,
                   grounding=v["grounding"], evolution=v["evolution"],
                   review=v["review"], groups=v["groups"], seeds=v["seeds"],
                   llm=v["llm"])
        if len(p):
            ics = p["ic"].to_numpy()
            row.update(preq_mean=ics.mean(),
                       preq_se=ics.std(ddof=1) / math.sqrt(len(ics)),
                       preq_hit=float((ics > 0).mean()),
                       preq_blocks=len(ics))
        paths = pit_paths(run)
        for kind in ("book", "pool"):
            if kind in paths:
                lz = read_pit(paths[kind])
                if len(lz):
                    ics = lz["ic"].dropna().to_numpy()
                    row.update({
                        f"lasso_{kind}_mean": ics.mean(),
                        f"lasso_{kind}_se":
                            ics.std(ddof=1) / math.sqrt(len(ics)),
                        f"lasso_{kind}_hit": float((ics > 0).mean()),
                        f"lasso_{kind}_n_avail":
                            float(lz["n_avail"].mean()),
                        f"lasso_{kind}_n_selected":
                            float(lz["n_selected"].dropna().mean())
                            if lz["n_selected"].notna().any() else np.nan,
                    })
        row.update(n_book=st.get("n_archive"), n_pool=st.get("n_pool"),
                   n_trials=st.get("n_trials"),
                   cost_usd=cost, flip_share=phi, flip_n=n_phi,
                   mean_abs_corr=div.get("mean_abs_corr"),
                   n_eff=div.get("effective_n_participation_ratio"),
                   n_book_analysed=div.get("n_factors"))
        rows.append(row)
    ladder = pd.DataFrame(rows)
    if {"lasso_book_n_selected", "lasso_book_n_avail"} <= set(ladder.columns):
        ladder["lasso_book_sel_share"] = (
            ladder["lasso_book_n_selected"] / ladder["lasso_book_n_avail"])
    ladder.to_csv(OUT / "ladder_summary.csv", index=False)

    # 4b. best-method PIT columns (the earlier reports' headline numbers) ---
    if len(pit_all):
        pm = (pit_all.dropna(subset=["ic"])
              .groupby(["run", "book", "method"])["ic"]
              .agg(["mean", "count"]).reset_index())
        pm = pm[pm["count"] >= 9]
        for kind in ("book", "pool"):
            best = (pm[pm["book"] == kind]
                    .sort_values("mean", ascending=False)
                    .groupby("run").first().reset_index())
            ladder = ladder.merge(
                best[["run", "mean", "method"]].rename(columns={
                    "mean": f"pit_best_{kind}_mean",
                    "method": f"pit_best_{kind}_method"}),
                on="run", how="left")
        ladder.to_csv(OUT / "ladder_summary.csv", index=False)

    # 4c. reference arms: the 1-group decomposition variants LDU/LDP --------
    ref_rows = []
    for run, note in (("LDU_terra_s0", "1x4 decomposition variant of arm 1"),
                      ("LDP_terra_s0", "1x4 decomposition variant of arm 2")):
        row = dict(run=run, note=note)
        prq = SERVER / run / "prequential_record.csv"
        if prq.exists():
            p = pd.read_csv(prq)
            p = p[p["generation"] >= 11]
            ics = p["combined_oos_ic"].to_numpy()
            row.update(preq_mean=ics.mean(),
                       preq_se=ics.std(ddof=1) / math.sqrt(len(ics)),
                       preq_hit=float((ics > 0).mean()))
        race = SERVER / f"pit_combiners/{run}.jsonl"
        if race.exists():
            pr = read_pit_all_methods(race, run, "pool")
            g = (pr.dropna(subset=["ic"]).groupby("method")["ic"]
                 .agg(["mean", "count"]))
            g = g[g["count"] >= 9]
            if "lasso" in g.index:
                row["lasso_pool_mean"] = g.loc["lasso", "mean"]
            if len(g):
                row["pit_best_pool_mean"] = g["mean"].max()
                row["pit_best_pool_method"] = g["mean"].idxmax()
            row["lasso_pool_n_avail"] = float(pr["n_avail"].mean())
        ref_rows.append(row)
    pd.DataFrame(ref_rows).to_csv(OUT / "reference_arms.csv", index=False)

    # 5. costs by role -------------------------------------------------------
    crows = []
    for k, v in ARMS.items():
        total, roles = read_cost(v["run"])
        for role, c in roles.items():
            crows.append(dict(arm=k, name=v["name"], role=role, cost_usd=c))
    pd.DataFrame(crows).to_csv(OUT / "costs_by_role.csv", index=False)

    # 6. per-factor stats, all arms pooled ----------------------------------
    pf = []
    for k, v in ARMS.items():
        d = analysis_dir(v["run"])
        if d is None:
            continue
        df = pd.read_csv(d / "per_factor_blocks.csv")
        df.insert(0, "arm", k)
        df.insert(1, "name", v["name"])
        pf.append(df)
    if pf:
        pd.concat(pf, ignore_index=True).to_csv(
            OUT / "per_factor_all.csv", index=False)

    # 7. KG campaign ---------------------------------------------------------
    kg = REPO / "data/kg_campaign_local/results.csv"
    if kg.exists():
        pd.read_csv(kg).to_csv(OUT / "kg_campaign.csv", index=False)

    # 8. model-quality (4o-mini) early/late blocks --------------------------
    mq = preq[preq["run"].isin(["L1HB_terra_s0", "L1HB_4omini_s0"])].copy()
    mq["phase"] = np.where(mq["block"] <= 4, "early(b1-4)", "late(b5-10)")
    mq.to_csv(OUT / "model_quality_blocks.csv", index=False)

    # 8b. reconciliation vs the earlier v2 book-analysis ladder -------------
    v2p = REPO / "data/comparisons/wf_book_analysis/derived/ladder_summary.csv"
    if v2p.exists():
        v2 = pd.read_csv(v2p).set_index("arm")
        rec = []
        for _, r in ladder.iterrows():
            if r["run"] not in v2.index:
                continue
            o = v2.loc[r["run"]]
            best = r.get("pit_best_pool_mean")
            if pd.isna(best):
                best = r.get("pit_best_book_mean")
            rec.append(dict(
                run=r["run"],
                preq_mean_now=r.get("preq_mean"),
                preq_mean_v2=o.get("preq_mean_ic"),
                pit_best_now=best,
                pit_best_v2=o.get("pit_best"),
                pit_best_v2_method=o.get("pit_best_method")))
        pd.DataFrame(rec).to_csv(OUT / "reconciliation_vs_v2.csv",
                                 index=False)

    # 9. LaTeX main ladder table --------------------------------------------
    def fmt(x, nd=4):
        return "--" if x is None or (isinstance(x, float) and
                                     not np.isfinite(x)) else f"{x:.{nd}f}"

    lines = [
        r"\begin{tabular}{|c|l|r|r|c|r|r|r|r|r|r|r|}",
        r"\hline",
        r"\# & Arm & $\overline{\mathrm{IC}}^{\mathrm{preq}}$ & "
        r"$s/\sqrt{10}$ & $H$ & Lasso & $\Phi$ & $|\mathcal B|$ & "
        r"$N_{\mathrm{eff}}$ & $\bar{|\rho|}$ & $N_{\mathrm{tr}}$ & "
        r"Cost (\$) \\",
        r"\hline",
    ]
    for _, r in ladder.iterrows():
        if r["arm"] == "MQ":
            continue
        lines.append(
            f"{r['arm']} & \\texttt{{{r['name']}}} & "
            f"{fmt(r.get('preq_mean'))} & {fmt(r.get('preq_se'))} & "
            f"{fmt(r.get('preq_hit'), 1)} & "
            f"{fmt(r.get('lasso_book_mean'))} & "
            f"{fmt(r.get('flip_share'), 2)} & "
            f"{int(r['n_book']) if pd.notna(r.get('n_book')) else '--'} & "
            f"{fmt(r.get('n_eff'), 1)} & {fmt(r.get('mean_abs_corr'), 2)} & "
            f"{int(r['n_trials']) if pd.notna(r.get('n_trials')) else '--'} & "
            f"{fmt(r.get('cost_usd'), 2)} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (TEX / "ladder_main.tex").write_text("\n".join(lines) + "\n")

    # 10. LaTeX: grounding 2x2 ----------------------------------------------
    def cell(arm):
        r = ladder[ladder["arm"] == arm].iloc[0]
        pool = (f", pool Lasso {r['lasso_pool_mean']:.3f}"
                if pd.notna(r.get("lasso_pool_mean")) else "")
        return (f"\\texttt{{{r['name']}}}: {r['preq_mean']:.4f} "
                f"$\\pm$ {r['preq_se']:.4f} (hit {r['preq_hit']:.0%}{pool})"
                .replace("%", "\\%"))

    (TEX / "grounding_2x2.tex").write_text("\n".join([
        r"\begin{tabular}{|l|c|c|}", r"\hline",
        r" & no papers & papers \\", r"\hline",
        f"no graph & {cell('1')} & {cell('2')} \\\\",
        f"graph briefs & {cell('3')} & {cell('4')} \\\\",
        r"\hline", r"\end{tabular}"]) + "\n")

    # 11. LaTeX: pool vs curated book ---------------------------------------
    rows_t = []
    for _, r in ladder.iterrows():
        if pd.isna(r.get("lasso_pool_mean")) and pd.isna(
                r.get("lasso_book_mean")):
            continue
        def f(v, nd=4):
            return "--" if pd.isna(v) else f"{v:.{nd}f}"
        rows_t.append(
            f"{r['arm']} & \\texttt{{{r['name']}}} & "
            f"{f(r.get('lasso_book_mean'))} & "
            f"{'--' if pd.isna(r.get('lasso_book_n_avail')) else int(r['lasso_book_n_avail'])} & "
            f"{f(r.get('lasso_pool_mean'))} & "
            f"{'--' if pd.isna(r.get('lasso_pool_n_avail')) else int(r['lasso_pool_n_avail'])} \\\\")
    (TEX / "pool_vs_book.tex").write_text("\n".join(
        [r"\begin{tabular}{|c|l|r|r|r|r|}", r"\hline",
         r"\# & Arm & Lasso (book) & $n$ & Lasso (pool) & $n$ \\",
         r"\hline"] + rows_t + [r"\hline", r"\end{tabular}"]) + "\n")

    # 12. model-quality early/late summary ----------------------------------
    mq_rows = []
    for run, lab in (("L1HB_terra_s0", "GPT-5.6 Terra"),
                     ("L1HB_4omini_s0", "GPT-4o-mini")):
        p = preq[preq["run"] == run]
        early = p[p["block"] <= 4]["ic"]
        late = p[p["block"] >= 5]["ic"]
        mq_rows.append(dict(model=lab,
                            early_mean=early.mean(), late_mean=late.mean(),
                            delta=late.mean() - early.mean(),
                            full_mean=p["ic"].mean()))
    pd.DataFrame(mq_rows).to_csv(OUT / "model_quality_summary.csv",
                                 index=False)

    print(ladder[["arm", "name", "preq_mean", "preq_se", "preq_hit",
                  "lasso_book_mean", "lasso_pool_mean"
                  if "lasso_pool_mean" in ladder.columns else "preq_hit",
                  "flip_share", "n_book", "n_pool", "n_eff", "n_trials",
                  "cost_usd"]].to_string(index=False))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
