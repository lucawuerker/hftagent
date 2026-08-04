#!/usr/bin/env python
"""Walk-forward persona strategies through the deterministic strategy compiler.

For each persona (personas.yaml): filter the book by theme, fit the blended
RF+GBM combiner per walk-forward segment (book@gen g trades the block revealed
at gen g+1 — identical snapshots to scripts/prequential_deployment.py), compile
positions through the risk pipeline (smoothing, neutralisation, max-positions
hysteresis, no-trade band, vol targeting) and stitch the honest net-return
track.  Design: docs/research-evolution/FACTOR_TO_STRATEGY_DESIGN.md.

    ./venv/bin/python scripts/compile_strategy.py --prerun L2WF_terra_s0 \
        --config quant.config.nasdaq100_2010_wf.yaml
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("compile_strategy")

H = 6


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prerun", required=True)
    p.add_argument("--config", default="quant.config.nasdaq100_2010_wf.yaml")
    p.add_argument("--model", default="rf_gbm",
                   help="Combiner blend: rf_gbm (default) | gbm | rf")
    p.add_argument("--personas", default=None,
                   help="Comma list of persona keys (default: all in personas.yaml)")
    p.add_argument("--personas-file", default=None)
    p.add_argument("--snap-gens", default=None,
                   help="Book-snapshot generations (default: auto from run config)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    os.environ["QF_CONFIG_FILE"] = args.config
    os.environ.setdefault("QF_USE_MCP", "0")

    import numpy as np
    import pandas as pd
    import types

    from prequential_deployment import replay_snapshots
    from backtest_combined_book import PANEL_END, ann_stats
    from quant_fund_agent.agents.factor_research.evolution.progressive import (
        build_schedule,
    )
    from quant_fund_agent.backtesting.strategy_compiler import (
        MODEL_BLENDS,
        blend_predictions,
        compile_positions,
        load_personas,
        modal_sector_labels,
        strategy_returns,
    )
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import _combined_prediction
    from simulate_user_strategies import select_factors

    blend = MODEL_BLENDS[args.model]
    personas = load_personas(args.personas_file)
    if args.personas:
        keys = set(args.personas.split(","))
        personas = [q for q in personas if q.key in keys]

    prerun_dir = (REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit"
                  / "preruns" / args.prerun)
    evo_dir = prerun_dir / "evolution"
    out_dir = Path(args.out) if args.out else prerun_dir / "persona_strategies"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = json.loads((evo_dir / "run_config.json").read_text())
    wf_mode = bool(rc.get("wf_blocks"))
    if args.snap_gens:
        snap_gens = tuple(int(x) for x in args.snap_gens.split(","))
    elif wf_mode:
        snap_gens = tuple(range(rc["generations"] - rc["wf_blocks"],
                                rc["generations"]))
    else:
        snap_gens = (10, 12, 14, 16, 18)
    snapshots = replay_snapshots(evo_dir, snap_gens=snap_gens)

    code_by_fid = {}
    st = json.loads((evo_dir / "state.json").read_text())
    for eg in st["kept_pool"]:
        prog = eg["genome"]["programs"][0]
        code_by_fid[prog["factor_id"]] = prog["code"]
    fid_by_gid = {}
    with (evo_dir / "lineage.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") is None:
                fid_by_gid[r["genome_id"]] = r["factor_ids"][0]

    # factor-metadata rows for the persona theme filter
    fdb = json.loads((prerun_dir / "factors" / "factor_db.json").read_text())
    recs = fdb if isinstance(fdb, list) else (fdb.get("factors") or [])
    meta_by_fid = {}
    for r in recs:
        if not isinstance(r, dict) or "id" not in r:
            continue
        meta_by_fid[r["id"]] = {
            "factor_id": r["id"], "name": r.get("name", ""),
            "category": r.get("category") or "other",
            "trading_idea": " ".join([str(r.get("trading_idea") or ""),
                                      str(r.get("description") or "")]),
            "mechanism": str((r.get("metadata") or {}).get("mechanism", "")),
            "score": (r.get("backtest_metrics") or {}).get("ic_mean") or 0.0,
        }

    # ── panel, schedule, segments (identical frames to prequential_deployment) ──
    fields = sorted(usable_fields())
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    sector_labels = (modal_sector_labels(panel["sector"])
                     if "sector" in panel else None)
    forward_start = pd.Timestamp(PANEL_END)
    run_idx = idx if wf_mode else idx[idx < forward_start]
    schedule = build_schedule(
        run_idx, generations=rc["generations"], test_frac=rc["test_frac"],
        seed_frac=rc["seed_frac"], reveal_every=rc["reveal_every"],
        val_blocks=rc["val_blocks"], holdout_last=rc.get("final_holdout", False),
        wf_blocks=rc.get("wf_blocks", 0), wf_block_bars=rc.get("wf_block_bars", 126))

    segments = []
    for sg in snap_gens:
        b = schedule[sg + 1].block_bounds
        segments.append((sg, pd.Timestamp(b[0]), pd.Timestamp(b[0]),
                         pd.Timestamp(b[1]), f"block@gen{sg + 1}"))
    if not wf_mode:
        test_start_ts = pd.Timestamp(schedule[rc["generations"]].val_end_ts)
        segments.append((20, test_start_ts, test_start_ts, forward_start, "test_tail"))
        segments.append((20, forward_start, forward_start,
                         idx[-1] + pd.Timedelta(days=1), "forward_reserve"))

    # ── signals for the union of snapshot books ──
    discover_factors()
    union_fids = sorted({fid_by_gid[g] for s in snapshots.values() for g in s})
    log.info("computing %d unique factor signals", len(union_fids))
    sigs = {}
    for fid in union_fids:
        try:
            cls = get_factor_class(fid) or compile_factor(code_by_fid[fid], fid)
            sigs[fid] = compute_signal(cls, panel)
        except Exception as e:  # noqa: BLE001
            log.warning("factor %s failed (%s) — excluded", fid, e)

    class _Cfg(types.SimpleNamespace):
        def fast_model_params(self, model):
            # keep the RF affordable at ~1M training rows × many fits
            return {"n_estimators": 150} if model == "random_forest" else None

    cfg = _Cfg(target_horizon=H, fit_standardize="per_underlying")
    pred_cache: dict[tuple, object] = {}

    def blended_pred(book_fids: list[str], fit_end: pd.Timestamp):
        feats = [sigs[f] for f in book_fids if f in sigs]
        if len(feats) < 2:
            return None
        key_ids = tuple(sorted(f for f in book_fids if f in sigs))
        fit_mask = np.asarray(idx < fit_end)
        preds = {}
        for m in blend:
            ck = (key_ids, str(fit_end), m)
            if ck not in pred_cache:
                pred_cache[ck] = _combined_prediction(feats, close, fit_mask, cfg, m)
            if pred_cache[ck] is None:
                return None
            preds[m] = pred_cache[ck]
        return blend_predictions(preds, blend)

    # ── equal-weight buy-and-hold benchmark over the union of eval windows ──
    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)
    mkt = fwd1.mean(axis=1).iloc[:-1]
    ev_union = np.zeros(len(mkt), dtype=bool)
    for _sg, _fe, ev_start, ev_end, _tag in segments:
        ev_union |= np.asarray((mkt.index >= ev_start) & (mkt.index < ev_end))
    bench = ann_stats(mkt[ev_union])

    # ── per persona: theme filter → per-segment fit → compile → stitch ──
    report = {"prerun": args.prerun, "model": args.model,
              "snap_gens": list(snap_gens), "benchmark_ew": bench,
              "personas": {}}
    for q in personas:
        stitched, seg_rows = [], []
        for sg, fit_end, ev_start, ev_end, tag in segments:
            book = [fid_by_gid[g] for g in snapshots[sg]]
            # snapshot books contain members that were later evicted and never
            # persisted to the factor DB — default their metadata so the theme
            # filter (and the full-book fallback) always sees the whole book
            rows = [meta_by_fid.get(f) or
                    {"factor_id": f, "category": "other", "trading_idea": "",
                     "name": f, "mechanism": "", "score": 0.0}
                    for f in book]
            themed = [r["factor_id"] for r in
                      select_factors({"factors": rows}, q.theme())]
            if len(themed) < 3:
                log.info("[%s|%s] theme matched %d — falling back to full book",
                         q.key, tag, len(themed))
                themed = list(book)
            pred = blended_pred(themed, fit_end)
            if pred is None:
                log.warning("[%s|%s] no prediction — skipped", q.key, tag)
                continue
            weights = compile_positions(pred, close, q.risk,
                                        sector_labels=sector_labels)
            net, turnover, pnl = strategy_returns(weights, close)
            ev = (net.index >= ev_start) & (net.index < ev_end)
            seg_net = net[ev]
            stitched.append(seg_net)
            stats = ann_stats(seg_net)
            w_al = weights.iloc[:-1]        # align to net's index (last bar dropped)
            npos = int((w_al[ev] != 0).sum(axis=1).mean()) if ev.any() else 0
            seg_rows.append({
                "segment": tag, "book_generation": sg, "n_factors": len(themed),
                "sharpe": stats["sharpe"], "ann_ret": stats["ann_ret"],
                "ann_vol": stats["ann_vol"],
                "turnover": float(turnover[ev].mean()),
                "mean_positions": npos,
            })
            log.info("[%s|%s] %d factors  sharpe=%s  turn=%.3f  pos=%d",
                     q.key, tag, len(themed),
                     None if stats["sharpe"] is None else f"{stats['sharpe']:.2f}",
                     float(turnover[ev].mean()), npos)
        ser = (pd.concat(stitched).sort_index() if stitched else pd.Series(dtype=float))
        ser = ser[~ser.index.duplicated()]
        ser.to_csv(out_dir / f"net_{q.key}.csv")
        full = ann_stats(ser)
        full["mean_turnover"] = float(np.mean([r["turnover"] for r in seg_rows])) \
            if seg_rows else None
        full["mean_positions"] = float(np.mean([r["mean_positions"] for r in seg_rows])) \
            if seg_rows else None
        report["personas"][q.key] = {"risk": q.risk.__dict__,
                                     "stitched": full, "segments": seg_rows}
        log.info("[%s] STITCHED sharpe=%s ann_ret=%s vol=%s",
                 q.key, full.get("sharpe"), full.get("ann_ret"), full.get("ann_vol"))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [f"# Persona strategies — {args.prerun} ({args.model})", "",
             f"Benchmark (equal-weight universe, buy & hold, no costs): "
             f"sharpe {bench.get('sharpe'):.2f}, ann ret "
             f"{(bench.get('ann_ret') or 0) * 100:.1f}%, vol "
             f"{(bench.get('ann_vol') or 0) * 100:.1f}%", "",
             "| persona | sharpe | ann ret | ann vol | max dd | turnover/d | ~positions |",
             "|---|---|---|---|---|---|---|"]
    for k, v in report["personas"].items():
        s = v["stitched"]
        def _f(x, pct=False):
            if x is None:
                return "–"
            return f"{x * 100:.1f}%" if pct else f"{x:.2f}"
        lines.append(f"| {k} | {_f(s.get('sharpe'))} | {_f(s.get('ann_ret'), True)} "
                     f"| {_f(s.get('ann_vol'), True)} | {_f(s.get('max_dd'), True)} "
                     f"| {_f(s.get('mean_turnover'))} | {_f(s.get('mean_positions'))} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
