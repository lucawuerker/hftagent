"""Prequential deployment simulation of an evolutionary run.

Simulates deploying the evolving factor book in real time: after the first 10
generations ("classical procedure"), the book as of generation g is traded on
the NEXT progressive-reveal block — data no selection pressure (and no combiner
fit) had seen at that point. After each reveal the book is updated to the new
front, the combiner refit on the now-visible window, and the next block traded.
The stitched result is a genuinely out-of-sample IC / return track across the
second half of the run, ending with the final book on the never-revealed
in-panel TEST tail and the 2-year forward reserve.

Book snapshots are reconstructed by replaying lineage.jsonl through the exact
controller semantics (non-dominated re-front on insert, batch rescore + prune
at reveals); the replay is verified against every logged eviction and the
final state archive before anything is traded.

The combiner is FIXED to lightgbm a priori (the winner of the earlier
combined-book race) — no model race is run against these blocks, so no new
selection touches them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prequential_deployment")

MODEL = "lightgbm"
SNAP_GENS = (10, 12, 14, 16, 18)   # book snapshots; final book (gen 20) added


# ── lineage replay: archive membership per generation ───────────────────────

def replay_snapshots(evo_dir: Path, snap_gens=SNAP_GENS) -> dict[int, list[str]]:
    """Book (genome_ids) as of the END of each snapshot generation, verified."""
    from collections import Counter

    from quant_fund_agent.research_eval.fitness import (
        GateResults,
        ObjectiveVector,
        dominates,
    )

    rows = [json.loads(l) for l in (evo_dir / "lineage.jsonl").open()]
    # Per mechanism-group archives: dominance/pruning happen strictly WITHIN a
    # group (the accepted book is their union), so the replay partitions too —
    # a single-group run (e.g. L2) degenerates to the old global behaviour.
    archives: dict[int, dict[str, tuple]] = {}
    group_of: dict[str, int] = {}
    my_evicts, logged_evicts = [], []
    snapshots: dict[int, set[str]] = {}
    pending: dict[str, tuple] = {}

    def union_ids() -> set[str]:
        return {gid for grp in archives.values() for gid in grp}

    def apply_pending():
        nonlocal pending
        if not pending:
            return
        for gid, val in pending.items():
            grp = archives.get(group_of.get(gid, -1), {})
            if gid in grp:
                grp[gid] = val
        for grp in archives.values():
            for gid in list(grp):
                if not grp[gid][1].passed:
                    del grp[gid]
                    my_evicts.append((gid, "rescore_gate_fail"))
            items = list(grp.items())
            for gid, (obj, _) in items:
                if any(dominates(o2, obj) for g2, (o2, _) in items if g2 != gid):
                    del grp[gid]
                    my_evicts.append((gid, "rescore_dominated"))
        pending = {}

    def maybe_snapshot(next_gen: int):
        for sg in snap_gens:
            if sg < next_gen and sg not in snapshots:
                snapshots[sg] = union_ids()

    for r in rows:
        ev = r.get("event")
        if ev == "rescore":
            maybe_snapshot(r["generation"])
            pending[r["genome_id"]] = (
                ObjectiveVector.from_dict(r["objective_after"]),
                GateResults.from_dict(r["gates_after"]))
            continue
        if ev == "archive_evict":
            logged_evicts.append((r["genome_id"], r["reason"]))
            continue
        apply_pending()
        maybe_snapshot(r["generation"])
        if not r.get("gates", {}).get("passed"):
            continue
        gid = r["genome_id"]
        gnum = int(r.get("mechanism_group_id") or 0)
        group_of[gid] = gnum
        grp = archives.setdefault(gnum, {})
        cand = (ObjectiveVector.from_dict(r["objective"]),
                GateResults.from_dict(r["gates"]))
        pool = list(grp.items()) + [(gid, cand)]
        new = {g1: v1 for g1, v1 in pool
               if not any(dominates(o2, v1[0])
                          for g2, (o2, _) in pool if g2 != g1)}
        for g1, _ in pool[:-1]:
            if g1 in grp and g1 not in new:
                my_evicts.append((g1, "dominated"))
        archives[gnum] = new
    apply_pending()
    maybe_snapshot(10**9)

    st = json.loads((evo_dir / "state.json").read_text())
    final_ids = {e["genome"]["genome_id"]
                 for grp in st["group_archives"] for e in grp}
    if union_ids() != final_ids:
        raise RuntimeError("lineage replay does not reproduce the final archive")
    if Counter(my_evicts) != Counter(logged_evicts):
        raise RuntimeError("lineage replay evictions do not match the log")
    log.info("replay VERIFIED: final archive %d/%d, evictions %d/%d match",
             len(union_ids()), len(final_ids), len(my_evicts), len(logged_evicts))
    out = {g: sorted(s) for g, s in snapshots.items()}
    out[20] = sorted(union_ids())
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="quant.config.nasdaq100_2010_forward.yaml")
    p.add_argument("--prerun", default="L2_opus5_s0")
    p.add_argument("--out", default=None)
    p.add_argument("--snap-gens", default=None,
                   help="Comma list of book-snapshot generations (default "
                        f"{','.join(map(str, SNAP_GENS))}); each book@gen g "
                        "trades the block revealed at gen g+1, so earlier "
                        "gens extend the walk-forward further back.")
    args = p.parse_args()
    snap_gens = (tuple(int(x) for x in args.snap_gens.split(","))
                 if args.snap_gens else SNAP_GENS)

    os.environ["QF_CONFIG_FILE"] = args.config
    os.environ.setdefault("QF_USE_MCP", "0")

    import numpy as np
    import pandas as pd
    import types

    from backtest_combined_book import (
        CONSTRUCTIONS,
        H,
        PANEL_END,
        ann_stats,
    )
    from quant_fund_agent.agents.factor_research.evolution.progressive import (
        build_schedule,
    )
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import (
        _combined_prediction,
        _pooled_ic,
    )

    prerun_dir = (REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit"
                  / "preruns" / args.prerun)
    evo_dir = prerun_dir / "evolution"
    out_dir = Path(args.out) if args.out else prerun_dir / "prequential_deployment"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = json.loads((evo_dir / "run_config.json").read_text())
    snapshots = replay_snapshots(evo_dir, snap_gens=snap_gens)

    # genome_id -> (factor_id, code) via lineage + kept_pool
    st = json.loads((evo_dir / "state.json").read_text())
    code_by_fid = {eg["genome"]["programs"][0]["factor_id"]:
                   eg["genome"]["programs"][0]["code"]
                   for eg in st["kept_pool"]}
    fid_by_gid = {}
    with (evo_dir / "lineage.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") is None:
                fid_by_gid[r["genome_id"]] = r["factor_ids"][0]

    # ── panel + reveal schedule (recomputed on the RUN panel index) ──
    fields = sorted(usable_fields())
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    # WF-ladder runs (wf_blocks > 0) have NO forward reserve and NO test tail:
    # the run panel is the full config panel and the walk-forward blocks reach
    # its edge (decision 2026-08-01).
    wf_mode = bool(rc.get("wf_blocks"))
    forward_start = pd.Timestamp(PANEL_END)
    run_idx = idx if wf_mode else idx[idx < forward_start]
    schedule = build_schedule(
        run_idx, generations=rc["generations"], test_frac=rc["test_frac"],
        seed_frac=rc["seed_frac"], reveal_every=rc["reveal_every"],
        val_blocks=rc["val_blocks"],
        holdout_last=rc.get("final_holdout", False),
        wf_blocks=rc.get("wf_blocks", 0),
        wf_block_bars=rc.get("wf_block_bars", 126))
    prog = json.loads((evo_dir / "progressive.json").read_text())
    assert schedule[rc["generations"]].visible_end == prog["visible_end"], \
        "recomputed schedule disagrees with the run's recorded frontier"
    test_start_ts = pd.Timestamp(schedule[rc["generations"]].val_end_ts)
    if wf_mode:
        # both extra frames are empty — the blocks cover everything
        forward_start = idx[-1] + pd.Timedelta(days=1)
        test_start_ts = forward_start
        log.info("schedule verified (wf mode): dev frontier reaches the panel "
                 "edge %s (bar %d); no TEST tail, no forward reserve",
                 idx[-1].date(), prog["visible_end"])
    else:
        log.info("schedule verified: dev frontier %s (bar %d), TEST tail %s -> %s, "
                 "forward %s -> %s", test_start_ts.date(), prog["visible_end"],
                 test_start_ts.date(), forward_start.date(), forward_start.date(),
                 idx[-1].date())

    # segments: (snapshot_gen, fit_end_ts, eval_start_ts, eval_end_ts, tag)
    segments = []
    for sg in snap_gens:
        b = schedule[sg + 1].block_bounds       # block revealed at gen sg+1
        segments.append((sg, pd.Timestamp(b[0]), pd.Timestamp(b[0]),
                         pd.Timestamp(b[1]), f"block@gen{sg + 1}"))
    if not wf_mode:
        segments.append((20, test_start_ts, test_start_ts, forward_start, "test_tail"))
        segments.append((20, forward_start, forward_start,
                         idx[-1] + pd.Timedelta(days=1), "forward_reserve"))

    # ── compute signals once for the union of all snapshot books ──
    discover_factors()
    union_fids = sorted({fid_by_gid[g] for s in snapshots.values() for g in s})
    log.info("computing %d unique factor signals over the union of books",
             len(union_fids))
    sigs: dict[str, object] = {}
    for fid in union_fids:
        try:
            cls = get_factor_class(fid) or compile_factor(code_by_fid[fid], fid)
            sigs[fid] = compute_signal(cls, panel)
        except Exception as e:  # noqa: BLE001
            log.warning("factor %s failed (%s) — excluded", fid, e)

    # ── walk forward ──
    cfg = types.SimpleNamespace(target_horizon=H, fit_standardize="per_underlying")
    rows = []
    stitched = {c: [] for c in CONSTRUCTIONS}   # net-return segments
    stitched_pred = []                          # (eval-sliced) predictions
    for sg, fit_end, ev_start, ev_end, tag in segments:
        book = [fid_by_gid[g] for g in snapshots[sg]]
        seg_sigs = [sigs[f] for f in book if f in sigs]
        fit_mask = np.asarray(idx < fit_end)
        ev_mask = np.asarray((idx >= ev_start) & (idx < ev_end))
        pred = _combined_prediction(seg_sigs, close, fit_mask, cfg, MODEL)
        if pred is None:
            log.warning("[%s] no prediction — skipped", tag)
            continue
        ic = _pooled_ic(pred, close, H, ev_mask, fit_mask, fit_mask | ev_mask)[0]
        row = {"segment": tag, "book_generation": sg, "n_factors": len(seg_sigs),
               "fit_end": str(fit_end.date()), "eval_start": str(idx[ev_mask][0].date()),
               "eval_end": str(idx[ev_mask][-1].date()),
               "n_eval_bars": int(ev_mask.sum()), "pooled_ic": ic}
        for cname, fn in CONSTRUCTIONS.items():
            net, turnover, gross = fn(pred, close)
            seg_net = net[ev_mask[:-1]]
            stitched[cname].append(seg_net)
            stats = ann_stats(seg_net)
            row[f"{cname}_sharpe"] = stats["sharpe"]
            row[f"{cname}_ann_ret"] = stats["ann_ret"]
            row[f"{cname}_turnover"] = float(turnover[ev_mask[:-1]].mean())
        stitched_pred.append(pred.loc[ev_mask])
        rows.append(row)
        log.info("[%s] book@gen%d (%d factors)  fit<%s  eval %s->%s  "
                 "IC=%.4f  cs_sharpe=%s  pu_sharpe=%s", tag, sg,
                 len(seg_sigs), fit_end.date(), row["eval_start"],
                 row["eval_end"], ic if ic is not None else float("nan"),
                 f"{row['cross_sectional_sharpe']:.2f}" if row["cross_sectional_sharpe"] else None,
                 f"{row['per_underlying_sharpe']:.2f}" if row["per_underlying_sharpe"] else None)

    # ── stitch + persist ──
    report = {"prerun": args.prerun, "model": MODEL,
              "snapshots": {str(g): [fid_by_gid[x] for x in s]
                            for g, s in snapshots.items()},
              "segments": rows}
    for cname in CONSTRUCTIONS:
        ser = pd.concat(stitched[cname]).sort_index()
        ser = ser[~ser.index.duplicated()]
        ser.to_csv(out_dir / f"stitched_net_{cname}.csv")
        report[f"stitched_{cname}"] = ann_stats(ser)
        # sub-splits: within-run (reveal blocks) vs test vs forward — the
        # latter two don't exist for wf-mode runs (blocks cover everything)
        report[f"stitched_{cname}_blocks"] = ann_stats(ser[ser.index < test_start_ts])
        if not wf_mode:
            report[f"stitched_{cname}_test"] = ann_stats(
                ser[(ser.index >= test_start_ts) & (ser.index < forward_start)])
            report[f"stitched_{cname}_forward"] = ann_stats(ser[ser.index >= forward_start])
    pred_all = pd.concat(stitched_pred).sort_index()
    pred_all = pred_all[~pred_all.index.duplicated()]
    pred_all.to_csv(out_dir / "stitched_prediction.csv")
    # 63-day rolling per-name IC of the stitched prediction (honest track)
    fwd = close.shift(-H) / close - 1.0
    fwd = fwd.reindex(pred_all.index)
    rc63 = pred_all.rolling(63, min_periods=40).corr(fwd)
    roll = pd.DataFrame({"mean": rc63.mean(axis=1),
                         "q25": rc63.quantile(0.25, axis=1),
                         "q75": rc63.quantile(0.75, axis=1)})
    roll.to_csv(out_dir / "stitched_rolling_pu_ic.csv")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "snapshots"},
                     indent=2, default=str))
    print("\nDONE ->", out_dir)


if __name__ == "__main__":
    main()
