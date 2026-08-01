#!/usr/bin/env python
"""Exhaustive post-run analysis of an evolutionary factor-research prerun.

Reproduces (generalised) the ad-hoc L2_opus5_s0 analysis: reads the run
artifacts (state / lineage / gen_quality / prequential / llm_usage / factor DB),
re-fits the final book on the run's own final IS window and scores it on the
final VAL window through the ``score_book_oos`` seam (combined LightGBM + Ridge
IC, per-factor solo ICs, LOCO marginals), computes book diversity (pairwise
signal correlations, participation ratio), and writes

    <run_dir>/analysis_data.json      (figure-suite input, same schema as L2's)
    <run_dir>/analysis_report.md      (numeric sections; Key-findings prose is
                                       appended by hand afterwards)

Usage:
    QF_CONFIG_FILE=quant.config.nasdaq100_2010.yaml \
    ./venv/bin/python scripts/analyze_evolution_run.py --run L4_terra_s0 \
        [--workspace fmp_archive_equity_nasdaq100pit] [--title "L4 (…)"]

Heavy step: one full panel load + 2 + N score_book_oos fits (N = book size,
for the LOCO column) — a few minutes on the 209-ticker panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QF_USE_MCP", "0")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="prerun name (e.g. L4_terra_s0)")
    ap.add_argument("--workspace", default="fmp_archive_equity_nasdaq100pit")
    ap.add_argument("--title", default=None, help="report H1 title")
    ap.add_argument("--skip-panel", action="store_true",
                    help="only the artifact-derived sections (no panel fits)")
    args = ap.parse_args()

    run_dir = ROOT / "data/workspaces" / args.workspace / "preruns" / args.run
    evo = run_dir / "evolution"
    if not evo.exists():
        raise SystemExit(f"no evolution dir at {evo}")

    state = json.loads((evo / "state.json").read_text())
    run_cfg = json.loads((evo / "run_config.json").read_text())
    usage = json.loads((evo / "llm_usage.json").read_text())
    gen_quality = _jsonl(evo / "gen_quality.jsonl")
    prequential = [r for r in _jsonl(evo / "prequential.jsonl")]
    lineage = _jsonl(evo / "lineage.jsonl")

    # ── artifact-derived pieces ────────────────────────────────────────────────
    archive = []
    book: list[dict] = []          # [{"factor_id", "code"}] for score_book_oos
    for entry in state["archive"]:
        g, fit = entry["genome"], entry["fitness"]
        prog = g["programs"][0]
        archive.append({
            "factor_id": prog["factor_id"],
            "generation": g.get("generation"),
            "operator": g.get("operator"),
            "mechanism_group_id": g.get("mechanism_group_id"),
            "objective": fit.get("objective"),
            "diagnostics": fit.get("diagnostics"),
        })
        book.append({"factor_id": prog["factor_id"], "code": prog["code"]})

    billed = []
    for r in lineage:
        if "event" in r:
            continue
        obj = r.get("objective") or {}
        billed.append({
            "generation": r.get("generation"),
            "operator": r.get("operator"),
            "selectable": r.get("selectable"),
            "mechanism_group_id": r.get("mechanism_group_id"),
            "marginal_value": obj.get("marginal_value"),
            "independence": obj.get("independence"),
            "parsimony": obj.get("parsimony"),
            "structural_novelty": obj.get("structural_novelty"),
            "factor_id": (r.get("factor_ids") or ["?"])[0],
        })
    evictions_by_gen = dict(Counter(
        r["generation"] for r in lineage if r.get("event") == "archive_evict"))
    rescore_deltas = []
    for r in lineage:
        if r.get("event") != "rescore":
            continue
        try:
            delta = (r["objective_after"]["marginal_value"]
                     - r["objective_before"]["marginal_value"])
        except (KeyError, TypeError):
            continue
        rescore_deltas.append({"generation": r["generation"], "delta": delta})

    db = json.loads((run_dir / "factors" / "factor_db.json").read_text())
    recs = {r["id"]: r for r in db.get("factors", [])}
    factor_meta = {fid: {"category": recs[fid].get("category"),
                         "name": recs[fid].get("name")}
                   for fid in (b["factor_id"] for b in book) if fid in recs}

    data: dict = {
        "archive": archive, "gen_quality": gen_quality,
        "prequential": prequential, "billed": billed,
        "evictions_by_gen": evictions_by_gen,
        "rescore_deltas": rescore_deltas, "usage": usage,
        "factor_meta": factor_meta,
    }

    # ── panel-dependent pieces ─────────────────────────────────────────────────
    window_info = ""
    if not args.skip_panel:
        import numpy as np
        import pandas as pd

        from quant_fund_agent.agents.factor_research.evolution.progressive import (
            build_schedule,
        )
        from quant_fund_agent.comparison.standardize import per_underlying_zscore
        from quant_fund_agent.factors import inmem
        from quant_fund_agent.mcp import research_client, research_service as svc

        fields = sorted({f for r in recs.values()
                         for f in (r.get("required_inputs") or [])}
                        | {"open", "high", "low", "close", "volume"})
        n_tickers = run_cfg.get("n_tickers")
        data_dir = run_cfg.get("data_dir", "ticker_data")
        h = int(run_cfg.get("target_horizon", 6))

        tl = research_client.panel_timeline(
            data_dir=data_dir, n_tickers=n_tickers, fields=fields,
            cutoff_date=run_cfg.get("cutoff_date"))
        index = pd.to_datetime(list(tl["index"]))
        sched = build_schedule(
            index, generations=int(run_cfg["generations"]),
            test_frac=float(run_cfg.get("test_frac", 0.2)),
            seed_frac=float(run_cfg.get("seed_frac", 0.45)),
            reveal_every=int(run_cfg.get("reveal_every", 1)),
            val_blocks=int(run_cfg.get("val_blocks", 2)),
            holdout_last=bool(run_cfg.get("final_holdout", False)))
        final_w = sched[-1]
        is_end, val_end = final_w.is_end_ts, final_w.val_end_ts
        window_info = (
            f"dev window: {index[0].date()} -> "
            f"{pd.Timestamp(val_end).date()} ({final_w.visible_end} bars); "
            f"TEST tail untouched\n"
            f"IS bars={final_w.val_start} "
            f"VAL bars={final_w.visible_end - final_w.val_start}")

        common = dict(start=is_end, end=val_end, target_horizon=h,
                      data_dir=data_dir, n_tickers=n_tickers, fields=fields)
        lgb = research_client.score_book_oos(book, marginal_model="lightgbm",
                                             **common)
        ridge = research_client.score_book_oos(book, marginal_model="ridge",
                                               **common)
        if not lgb.get("ok"):
            raise SystemExit(f"score_book_oos failed: {lgb.get('error')}")
        data["combined"] = {"lightgbm_val_ic": lgb.get("combined_oos_ic"),
                            "ridge_val_ic": ridge.get("combined_oos_ic")}
        solo = lgb.get("per_factor_oos_ic") or {}

        per_factor_fit: dict[str, dict] = {}
        base_ic = lgb.get("combined_oos_ic")
        for i, member in enumerate(book):
            fid = member["factor_id"]
            rest = [b for b in book if b["factor_id"] != fid]
            r = research_client.score_book_oos(rest, marginal_model="lightgbm",
                                               **common)
            loco = (base_ic - r["combined_oos_ic"]) \
                if (r.get("ok") and base_ic is not None
                    and r.get("combined_oos_ic") is not None) else None
            per_factor_fit[fid] = {"solo_val_ic": solo.get(fid),
                                   "loco_marginal": loco}
            print(f"  LOCO {i + 1}/{len(book)} {fid}", file=sys.stderr)
        data["per_factor_fit"] = per_factor_fit

        # book diversity on the dev window (z-scored signals, pairwise corr)
        panel = svc._load_panel_cached(data_dir, fields, n_tickers=n_tickers)
        close = panel["close"]
        dev_mask = close.index < pd.Timestamp(val_end)
        cols = {}
        for member in book:
            fid = member["factor_id"]
            try:
                sig = inmem.signal_from_code(member["code"], fid, panel)
            except Exception as e:  # noqa: BLE001
                print(f"  corr: {fid} failed ({e}) — skipped", file=sys.stderr)
                continue
            z = per_underlying_zscore(
                sig.reindex(index=close.index, columns=close.columns))
            cols[fid] = z.loc[dev_mask].to_numpy(dtype=float).ravel()
        corr_df = pd.DataFrame(cols).corr()
        data["corr_matrix"] = {"ids": list(corr_df.columns),
                               "values": corr_df.round(4).values.tolist()}
        vals = corr_df.fillna(0.0).to_numpy()
        eig = np.linalg.eigvalsh(np.nan_to_num(vals))
        data["eigenvalues"] = [float(x) for x in eig]

    (run_dir / "analysis_data.json").write_text(
        json.dumps(data, indent=1, default=str))
    print(f"wrote {run_dir / 'analysis_data.json'}")

    # ── report ────────────────────────────────────────────────────────────────
    rep: list[str] = []
    title = args.title or f"{args.run} — exhaustive run analysis"
    from datetime import date
    rep.append(f"# {title}\n")
    rep.append(f"Generated {date.today()} from run artifacts (lineage, "
               "gen_quality, prequential, state, factor DB, usage).")
    rep.append("TEST tail untouched — all numbers are dev-window or honest "
               "prequential OOS.\n")
    rep.append("```")

    t = usage["total"]
    rep.append("=== 1. RUN OVERVIEW ===")
    rep.append(f"trials: {state['n_trials']} | kept_pool: "
               f"{len(state.get('kept_pool', []))} | final archive: {len(archive)}")
    rep.append("usage by role:")
    for role, u in usage["by_role"].items():
        rep.append(f"  {role:<12} calls={u['calls']:>4} in={u['input_tokens'] / 1e6:>5.2f}M "
                   f"out={u['output_tokens'] / 1e6:>5.2f}M ${u['cost_usd']:>7.2f} "
                   f"errors={u['errors']}")
    rep.append(f"  TOTAL        calls={t['calls']} ${t['cost_usd']:.2f} "
               f"errors={t['errors']}\n")

    rep.append("=== 2. OPERATOR MIX (billed candidates) ===")
    by_op: dict[str, list] = {}
    for b in billed:
        by_op.setdefault(b["operator"], []).append(b)
    import statistics
    for op, rows in sorted(by_op.items(), key=lambda kv: -len(kv[1])):
        mv = [r["marginal_value"] for r in rows if r["marginal_value"] is not None]
        sel = sum(1 for r in rows if r["selectable"])
        rep.append(f"  {op:<22} n={len(rows):>4} selectable={sel:>4} "
                   f"mean_marginal={statistics.mean(mv):+.4f} max={max(mv):+.4f}"
                   if mv else f"  {op:<22} n={len(rows):>4} selectable={sel:>4}")
    rep.append("")

    rep.append("=== 3. GENERATION TRAJECTORY ===")
    rep.append("  gen  billed  archive kept_pool mean_mv   max_mv   novelty evict")
    billed_by_gen = Counter(b["generation"] for b in billed)
    for gq in gen_quality:
        g = gq["generation"]
        rep.append(f"  {g:>3}  {billed_by_gen.get(g, 0):>5}  {gq['archive_size_total']:>6} "
                   f"{gq['kept_pool_size']:>8} {gq['mean_marginal_value']:+.4f}  "
                   f"{gq['max_marginal_value']:+.4f}  {gq['mean_structural_novelty']:.3f} "
                   f"{evictions_by_gen.get(g, 0):>4}")
    rep.append("")

    rep.append("=== 4. PREQUENTIAL (honest OOS on never-seen blocks) ===")
    rep.append("  idx gen  window                        combined_OOS_IC   PBO    n_obs archive")
    ics = []
    for r in prequential:
        if "skipped" in r:
            rep.append(f"    {r.get('reveal_index')}  {r['generation']:>2}  SKIPPED: {r['skipped']}")
            continue
        ic = r.get("combined_oos_ic")
        ics.append(ic)
        rep.append(f"    {r.get('reveal_index')}  {r['generation']:>3}  "
                   f"{r['start'][:10]} -> {r['end'][:10]}   {ic:+.4f}      "
                   f"{r.get('pbo')}    {r.get('n_obs')}   {r.get('archive_size')}")
    if ics:
        rep.append(f"  mean={statistics.mean(ics):+.4f} median={statistics.median(ics):+.4f} "
                   f"min={min(ics):+.4f} max={max(ics):+.4f} "
                   f"positive={sum(1 for x in ics if x > 0)}/{len(ics)}")
    rep.append("")

    rep.append(f"=== 5. FINAL ARCHIVE (the {len(archive)}-factor book) ===")
    rep.append("  factor_id                                          grp gen op            marginal indep   pars novelty")
    for a in sorted(archive, key=lambda a: -(a["objective"].get("marginal_value") or -9)):
        o = a["objective"]
        rep.append(f"  {a['factor_id'][:48]:<50} {a.get('mechanism_group_id')} "
                   f"{a['generation']:>3} {a['operator']:<12} "
                   f"{(o.get('marginal_value') or 0):+.4f} {(o.get('independence') or 0):+.4f} "
                   f"{int(o.get('parsimony') or 0):>5} {(o.get('structural_novelty') or 0):.3f}")
    ages = Counter(a["generation"] for a in archive)
    rep.append(f"  book age distribution (generation born): {dict(sorted(ages.items()))}")
    groups = Counter(a.get("mechanism_group_id") for a in archive)
    rep.append(f"  book mechanism-group distribution: {dict(sorted(groups.items()))}")
    cats = Counter((factor_meta.get(a['factor_id']) or {}).get('category')
                   for a in archive)
    rep.append(f"  categories: {dict(cats.most_common())}")
    rep.append("```\n")

    if not args.skip_panel:
        rep.append("```")
        rep.append(window_info + "\n")
        rep.append("=== COMBINED MODEL (fit IS, scored VAL) ===")
        rep.append(f"LightGBM combined VAL IC ({len(book)} factors): "
                   f"{data['combined']['lightgbm_val_ic']:+.4f}")
        rep.append(f"Ridge    combined VAL IC ({len(book)} factors): "
                   f"{data['combined']['ridge_val_ic']:+.4f}\n")
        rep.append("=== PER-FACTOR: solo VAL IC | LOCO marginal on final book ===")
        pf = data["per_factor_fit"]
        for fid, v in sorted(pf.items(),
                             key=lambda kv: -(kv[1]["loco_marginal"] or -9)):
            solo_s = f"{v['solo_val_ic']:+.4f}" if v["solo_val_ic"] is not None else "  n/a"
            loco_s = f"{v['loco_marginal']:+.4f}" if v["loco_marginal"] is not None else "  n/a"
            rep.append(f"  {fid[:50]:<52} solo={solo_s}  LOCO_marginal={loco_s}")
        rep.append("")
        rep.append("=== BOOK DIVERSITY (dev window) ===")
        import numpy as np
        ids = data["corr_matrix"]["ids"]
        m = np.array(data["corr_matrix"]["values"], dtype=float)
        off = m[~np.eye(len(ids), dtype=bool)]
        off = off[~np.isnan(off)]
        i, j = divmod(int(np.nanargmax(np.abs(np.triu(m, 1)))), len(ids))
        rep.append(f"mean |pairwise corr| = {np.nanmean(np.abs(off)):.3f} | "
                   f"median = {np.nanmedian(np.abs(off)):.3f} | max = {np.nanmax(np.abs(off)):.3f}")
        rep.append(f"most correlated pair: {ids[i][:40]} ~ {ids[j][:40]} "
                   f"(rho={m[i, j]:+.3f})")
        eig = np.array(data["eigenvalues"])
        pr = float(eig.sum() ** 2 / (eig ** 2).sum())
        rep.append(f"participation ratio (effective independent factors): "
                   f"{pr:.1f} / {len(ids)}\n")
        rep.append("=== ARCHIVE DIAGNOSTICS (recorded by the run) ===")
        rep.append("  factor_id                                            IS_IC   degrad  max|corr| coverage")
        for a in archive:
            d = a.get("diagnostics") or {}
            deg = d.get("degradation_ratio")
            deg_s = f"{deg:+.2f}" if isinstance(deg, (int, float)) else "  None"
            rep.append(f"  {a['factor_id'][:50]:<52} "
                       f"{(d.get('is_ic') or 0):+.3f}  {deg_s}  "
                       f"{(d.get('max_abs_corr') or 0):.3f}   {(d.get('coverage') or 0):.2f}")
        rep.append("```\n")

    (run_dir / "analysis_report.md").write_text("\n".join(rep))
    print(f"wrote {run_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
