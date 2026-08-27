"""KG-campaign IC worker (runs on lagias, asynchronously to the seeding chain).

Watches ``data/kg_campaign/queue/run_NN.ready``. For each completed seeding
run it computes the mean walk-forward IC (10 per-block refits, blocks =
the ladder's prequential windows, PIT-honest with full availability since
seeding never sees returns):

  * run book  : ridge + LassoCV (small)
  * cumulative: ridge always (Gram trick, alpha fixed at 1e4 — the centre of
                the CV-optimum band {3.2k..32k} observed across every book of
                the combiner campaign; a-priori choice, no per-run tuning);
                LassoCV only while the cumulative book has <= 500 factors.

Appends ``data/kg_campaign/results.csv`` + per-block rows to
``data/kg_campaign/results_blocks.jsonl``. Stop by touching
``data/kg_campaign/STOP``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kg_ic_worker")

CAMP = REPO / "data/kg_campaign"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
REF_PREQUENTIAL = WS / "L4WF_terra_s0/evolution/prequential.jsonl"
H = 6
RIDGE_ALPHA = 1e4
LASSO_MAX_N = 500


def main() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV

    from wf_common import load_or_compute_signal
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import (_label_available_mask,
                                                        _pooled_ic)

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    n_cols = close.shape[1]
    y_all = forward_returns(close, horizon=H).to_numpy(dtype=float).ravel()

    blocks = []
    for line in REF_PREQUENTIAL.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((r["generation"], pd.Timestamp(r["start"]),
                           pd.Timestamp(r["end"])))
    blocks.sort()

    def book_ics(codes: dict[str, str], methods: tuple[str, ...]):
        """{method: [per-block IC]} for one book under per-block refits."""
        sigs = {}
        for fid, code in codes.items():
            try:
                sigs[fid] = load_or_compute_signal(fid, code, panel, idx,
                                                   close.columns)
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed: %s", fid, e)
        fids = sorted(sigs)
        out = {m: [] for m in methods}
        n_rows = len(idx) * n_cols
        n_f = len(fids)
        # rows per float64 working chunk (~1.6 GB)
        chunk = max(50_000, int(2.0e8 // max(n_f, 1)))
        for g, start, end in blocks:
            fit_mask = np.asarray(idx < start)
            block_mask = np.asarray((idx >= start) & (idx < end))
            fit_idx = idx[fit_mask]
            # Fill a preallocated float32 design matrix column by column.  The
            # old list-of-columns + column_stack doubled peak RSS and, together
            # with a full float64 copy of X, OOM-killed the worker on run 18's
            # ~1.9k-factor cumulative book (2026-08-17, 31 GB anon-rss).
            X = np.empty((n_rows, n_f), dtype=np.float32)
            for j, f in enumerate(fids):
                z = per_underlying_zscore(sigs[f].astype(float), fit_idx)
                X[:, j] = np.nan_to_num(
                    z.to_numpy(dtype=float), nan=0.0, posinf=0.0,
                    neginf=0.0).ravel()
                del z
            lab_ok = _label_available_mask(fit_mask, H)
            fit_rows = np.repeat(np.asarray(fit_mask) & lab_ok, n_cols)
            train = np.flatnonzero(fit_rows & np.isfinite(y_all))
            yf = y_all[train]

            def predict_chunked(fn):
                """Apply fn to X in row chunks — never materialise X in f64."""
                flat = np.empty(n_rows, dtype=np.float64)
                for s in range(0, n_rows, chunk):
                    flat[s:s + chunk] = fn(
                        X[s:s + chunk].astype(np.float64))
                return flat.reshape(len(idx), n_cols)

            for m in methods:
                try:
                    if m == "ridge":
                        # Gram-trick ridge: (X'X + aI) w = X'y — N x N solve,
                        # scales to thousands of factors.  X'X and X'y are
                        # accumulated over row chunks (identical result, bounded
                        # memory).
                        G = np.zeros((n_f, n_f), dtype=np.float64)
                        Xty = np.zeros(n_f, dtype=np.float64)
                        for s in range(0, train.size, chunk):
                            sl = train[s:s + chunk]
                            Xc = X[sl].astype(np.float64)
                            G += Xc.T @ Xc
                            Xty += Xc.T @ yf[s:s + chunk]
                            del Xc
                        G[np.diag_indices_from(G)] += RIDGE_ALPHA
                        w = np.linalg.solve(G, Xty)
                        del G, Xty
                        pred = predict_chunked(lambda A, w=w: A @ w)
                    else:  # lasso (books <= LASSO_MAX_N factors only)
                        Xf = X[train].astype(np.float64)
                        est = LassoCV(cv=3, random_state=42,
                                      precompute=False)
                        est.fit(Xf, yf)
                        del Xf
                        pred = predict_chunked(est.predict)
                    pred_df = pd.DataFrame(pred, index=idx,
                                           columns=close.columns)
                    ic = _pooled_ic(pred_df, close, H, row_mask=block_mask,
                                    available_mask=block_mask)[0]
                    out[m].append(ic)
                    del pred, pred_df
                except Exception as e:  # noqa: BLE001
                    log.warning("g%d %s failed: %s", g, m, e)
                    out[m].append(None)
            del X
        return out, len(fids)

    def summarize(ics):
        ok = [i for i in ics if i is not None]
        if not ok:
            return None, None, None
        return (float(np.mean(ok)), float(np.std(ok)),
                float(np.mean([i > 0 for i in ok])))

    (CAMP / "queue").mkdir(parents=True, exist_ok=True)
    res_csv = CAMP / "results.csv"
    if not res_csv.exists():
        res_csv.write_text("run,scope,method,n_factors,blockmean,blockstd,"
                           "hit,n_blocks\n")
    done: set[str] = set()
    for line in res_csv.read_text().splitlines()[1:]:
        done.add(line.split(",")[0] + ":" + line.split(",")[1] + ":"
                 + line.split(",")[2])

    log.info("worker up: %d blocks, ridge alpha %g", len(blocks), RIDGE_ALPHA)
    while not (CAMP / "STOP").exists():
        ready = sorted((CAMP / "queue").glob("run_*.ready"))
        todo = None
        for r in ready:
            n = int(r.stem.split("_")[1])
            if f"{n}:cum:ridge" not in done:
                todo = n
                break
        if todo is None:
            time.sleep(60)
            continue
        n = todo
        log.info("processing run %02d", n)
        cum = json.loads((CAMP / "cumulative_book.json").read_text())
        cum_codes = {}
        missing = 0
        for e in cum:
            if e["run"] > n:
                continue
            p = Path(e["code_path"])
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                # cumulative_book paths are written on the seeding machine —
                # remap by basename onto this box's researcher package (the
                # chain rsyncs it every run)
                p = REPO / "quant_fund_agent/factors/researcher" / p.name
            if p.exists():
                cum_codes[e["factor_id"]] = p.read_text()
            else:
                missing += 1
        if missing:
            log.warning("run %02d: %d cumulative codes missing on this box",
                        n, missing)
        run_codes = {e["factor_id"]: cum_codes[e["factor_id"]]
                     for e in cum if e["run"] == n
                     and e["factor_id"] in cum_codes}

        for scope, codes in (("run", run_codes), ("cum", cum_codes)):
            if not codes:
                log.warning("run %02d scope=%s: no codes — skipped", n, scope)
                done.add(f"{n}:{scope}:ridge")
                continue
            methods = ["ridge"]
            if scope == "run" or len(codes) <= LASSO_MAX_N:
                methods.append("lasso")
            # a relaunch (crash / supervisor restart) picks the run back up by
            # its missing cum/ridge cell — don't recompute the scopes and
            # methods already in results.csv and append duplicate rows
            methods = [m for m in methods if f"{n}:{scope}:{m}" not in done]
            if not methods:
                continue
            try:
                ics, nf = book_ics(codes, tuple(methods))
            except Exception as e:  # noqa: BLE001
                log.warning("run %02d scope=%s FAILED: %s", n, scope, e)
                done.add(f"{n}:{scope}:ridge")
                continue
            for m in methods:
                bm, bs, hit = summarize(ics[m])
                if bm is None:
                    continue
                with res_csv.open("a") as fh:
                    fh.write(f"{n},{scope},{m},{nf},{bm:.6f},{bs:.6f},"
                             f"{hit:.2f},{sum(1 for i in ics[m] if i is not None)}\n")
                with (CAMP / "results_blocks.jsonl").open("a") as fh:
                    fh.write(json.dumps({"run": n, "scope": scope,
                                         "method": m, "ics": ics[m]}) + "\n")
                done.add(f"{n}:{scope}:{m}")
                log.info("run %02d %s/%s: n=%d mean=%.4f hit=%.0f%%",
                         n, scope, m, nf, bm, (hit or 0) * 100)
    log.info("STOP marker — worker exiting")


if __name__ == "__main__":
    main()
