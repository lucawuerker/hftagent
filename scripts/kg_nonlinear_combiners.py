"""Nonlinear combiner study on the KG-campaign clean pool (rho < 0.9).

Protocol = the standing PIT walk-forward convention (user decisions
2026-08-05 + 2026-08-17): the 10 prequential 126-bar blocks from 2021-07-20
(ladder reference windows); for each block every model is refit from scratch
on all bars strictly before the block start (expanding window) and scored on
the block via the harness's pooled per-underlying IC; the WF statistic is
the mean of the 10 block ICs.

Models (design signed off 2026-08-17):
  equal   sign(fit-IC)-aligned equal weights            (baseline)
  ic      weights proportional to fit-window IC          (baseline)
  ridge   Gram-trick ridge, alpha 1e4                    (linear baseline,
          = the kg_ic_worker convention, but on the clean pool)
  lightgbm gbdt with lambda_l2 = 5N / lambda_l1 = 50, feature_fraction 0.3,
          early stopping on a temporal tail-val pooled IC
  rf      LightGBM boosting_type=rf: true bagging (0.632) + per-node
          sqrt(N) feature subsampling, 300 deep trees
  nn      TabM-style parameter-efficient MLP ensemble (BatchEnsemble,
          k=8 members, 512-128-32, input dropout, Huber loss, AdamW,
          early stopping on tail-val pooled IC), PyTorch on MPS

Tree/NN models fit on the first 90% of the fit window's bars and early-stop
(rf: just fit) against the last 10%; the linear baselines fit on the full
fit window (the worker convention — favours the baselines if anything).

All models are saved under <out>/models/, per-block predictions under
<out>/preds/, results append to <out>/results.jsonl (resume-safe per
(model, block)).
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("kg_nonlinear")

import numpy as np
import pandas as pd

CAMP = REPO / "data/kg_campaign"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
REF_PREQUENTIAL = WS / "L4WF_terra_s0/evolution/prequential.jsonl"
OUT = REPO / "data/comparisons/kg_nonlinear_combiners"
H = 6
RIDGE_ALPHA = 1e4
VAL_FRAC = 0.10
SEED = 42
ALL_MODELS = ("lightgbm", "rf", "nn", "lasso")


# ── generic helpers ────────────────────────────────────────────────────────

def rows_ic(pred: np.ndarray, y: np.ndarray, tids: np.ndarray,
            n_assets: int) -> float | None:
    """Observation-weighted mean per-asset Pearson IC on flattened rows.

    Mirrors comparison.ic._weighted_asset_pearson (>=3 finite pairs per
    asset, zero-variance assets skipped, weight = pair count).
    """
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return None
    t, p, q = tids[m], pred[m].astype(np.float64), y[m].astype(np.float64)
    n = np.bincount(t, minlength=n_assets).astype(np.float64)
    sx = np.bincount(t, weights=p, minlength=n_assets)
    sy = np.bincount(t, weights=q, minlength=n_assets)
    sxx = np.bincount(t, weights=p * p, minlength=n_assets)
    syy = np.bincount(t, weights=q * q, minlength=n_assets)
    sxy = np.bincount(t, weights=p * q, minlength=n_assets)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = n * sxx - sx * sx
        vy = n * syy - sy * sy
        r = (n * sxy - sx * sy) / np.sqrt(vx * vy)
    ok = (n >= 3) & np.isfinite(r)
    if not ok.any():
        return None
    return float((r[ok] * n[ok]).sum() / n[ok].sum())


def block_ic(pred_blk: np.ndarray, blk_rows: np.ndarray, idx, cols,
             close, block_mask) -> float | None:
    """Score block predictions with the harness's own _pooled_ic."""
    from quant_fund_agent.research_eval.harness import _pooled_ic

    full = np.full(len(idx) * len(cols), np.nan)
    full[blk_rows] = pred_blk
    pred_df = pd.DataFrame(full.reshape(len(idx), len(cols)),
                           index=idx, columns=cols)
    ic, _n = _pooled_ic(pred_df, close, H, row_mask=block_mask,
                        available_mask=block_mask)
    return ic


# ── models ─────────────────────────────────────────────────────────────────

def fit_lightgbm(dtr, dval, val_tids, n_assets, mode: str,
                 n_feat: int, seed: int):
    """gbdt early-stops on the SMOOTH val l2 loss (the GKX/Qlib convention);
    an IC-based stop proved pathological (val IC is noisy iteration-to-
    iteration -> stopped at iteration 8 with a near-null model).  The val
    pooled IC is still reported post-fit."""
    import lightgbm as lgb

    common = dict(objective="regression", max_bin=127, seed=seed,
                  verbosity=-1)
    if mode == "gbdt":
        params = dict(common, boosting_type="gbdt", learning_rate=0.03,
                      num_leaves=63, min_data_in_leaf=500,
                      feature_fraction=0.3, bagging_fraction=0.7,
                      bagging_freq=1, lambda_l1=50.0,
                      lambda_l2=5.0 * n_feat, metric="l2")
        rounds, cbs = 1500, [lgb.early_stopping(75, verbose=False)]
    else:  # rf
        params = dict(common, boosting_type="rf", bagging_fraction=0.632,
                      bagging_freq=1, feature_fraction=1.0,
                      feature_fraction_bynode=max(0.01,
                                                  n_feat ** -0.5),
                      num_leaves=2047, min_data_in_leaf=250,
                      metric="None")
        rounds, cbs = 300, []
    booster = lgb.train(params, dtr, num_boost_round=rounds,
                        valid_sets=[dval], callbacks=cbs)
    return booster


def predict_chunked_lgb(booster, X16, chunk=100_000):
    out = np.empty(len(X16), dtype=np.float64)
    for a in range(0, len(X16), chunk):
        out[a:a + chunk] = booster.predict(
            np.asarray(X16[a:a + chunk], dtype=np.float32))
    return out


def fit_nn(X_tr16, y_tr, X_val16, y_val, val_tids, n_assets, seed: int,
           k: int = 8, max_epochs: int = 60, patience: int = 6,
           batch: int = 4096):
    """TabM-style BatchEnsemble MLP; returns (state, meta, predict_fn)."""
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)   # avoid the LightGBM/torch dual-libomp deadlock
    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_feat = X_tr16.shape[1]
    y_std = float(np.std(y_tr)) or 1.0

    class BELinear(nn.Module):
        def __init__(self, din, dout):
            super().__init__()
            self.W = nn.Parameter(torch.empty(din, dout))
            nn.init.kaiming_uniform_(self.W, a=5 ** 0.5)
            sign = torch.randint(0, 2, (k, din)).float() * 2 - 1
            self.r = nn.Parameter(sign)          # random-sign init (TabM)
            self.s = nn.Parameter(torch.ones(k, dout))
            self.b = nn.Parameter(torch.zeros(k, dout))

        def forward(self, x):                    # (B, k, din)
            return ((x * self.r) @ self.W) * self.s + self.b

    class TabM(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp_drop = nn.Dropout(0.15)
            self.l1, self.n1 = BELinear(n_feat, 512), nn.LayerNorm(512)
            self.l2, self.n2 = BELinear(512, 128), nn.LayerNorm(128)
            self.l3 = BELinear(128, 32)
            self.head = BELinear(32, 1)
            self.act, self.drop = nn.GELU(), nn.Dropout(0.25)

        def forward(self, x):                    # (B, n_feat)
            x = self.inp_drop(x)
            x = x.unsqueeze(1).expand(-1, k, -1)
            x = self.drop(self.act(self.n1(self.l1(x))))
            x = self.drop(self.act(self.n2(self.l2(x))))
            x = self.act(self.l3(x))
            return self.head(x).squeeze(-1)      # (B, k)

    model = TabM().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    y_scaled = (y_tr / y_std).astype(np.float32)

    def predict16(X16np, bs: int = 32768) -> np.ndarray:
        # numpy-side gather/cast: torch CPU indexing would enter an OpenMP
        # parallel region and deadlock against LightGBM's libomp copy
        model.eval()
        outs = []
        with torch.no_grad():
            for a in range(0, len(X16np), bs):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X16np[a:a + bs],
                                         dtype=np.float32)).to(dev)
                outs.append(model(xb).mean(dim=1).cpu().numpy())
        return np.concatenate(outs) if outs else np.empty(0)

    best_ic, best_state, best_epoch, bad = -np.inf, None, 0, 0
    n_tr = len(X_tr16)
    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n_tr)
        t_ep = time.time()
        for a in range(0, n_tr, batch):
            ridx = perm[a:a + batch]
            xb = torch.from_numpy(np.ascontiguousarray(X_tr16[ridx])).to(dev)
            yb = torch.from_numpy(y_scaled[ridx]).to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(xb)                      # (B, k)
            loss = loss_fn(out, yb.unsqueeze(1).expand(-1, k))
            loss.backward()
            opt.step()
        ic = rows_ic(predict16(X_val16), y_val, val_tids, n_assets)
        ic = -1.0 if ic is None else ic
        log.info("    nn epoch %d val_ic %.5f (%.0fs)", epoch, ic,
                 time.time() - t_ep)
        if ic > best_ic:
            best_ic, best_epoch, bad = ic, epoch, 0
            best_state = {kk: v.detach().cpu().clone()
                          for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    meta = dict(k=k, dims=[n_feat, 512, 128, 32], y_std=y_std,
                best_epoch=best_epoch, best_val_ic=best_ic, device=dev,
                seed=seed)
    return model, meta, predict16


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--blocks", type=int, default=0,
                    help="limit to first N blocks (0 = all)")
    ap.add_argument("--max-factors", type=int, default=0,
                    help="limit pool size (smoke tests)")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]

    from wf_common import SIGNAL_STORE
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import _label_available_mask

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", ["close"],
                                   n_tickers=None)
    close = panel["close"]
    idx, cols = close.index, close.columns
    n_bars, n_assets = len(idx), len(cols)
    y_all = forward_returns(close, horizon=H).to_numpy(
        dtype=np.float64).ravel()
    tids_all = np.tile(np.arange(n_assets), n_bars)

    book = json.loads((CAMP / "clean_book.json").read_text())
    if args.max_factors:
        book = book[: args.max_factors]
    fids = [b["factor_id"] for b in book]
    skeys = [b["signal_key"] for b in book]
    N = len(fids)
    log.info("clean pool: %d factors", N)

    blocks = []
    for line in REF_PREQUENTIAL.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((r["generation"], pd.Timestamp(r["start"]),
                           pd.Timestamp(r["end"])))
    blocks.sort()
    if args.blocks:
        blocks = blocks[: args.blocks]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["gen"]))
    (OUT / "factors.json").write_text(json.dumps(
        {"n": N, "factor_ids": fids}, indent=1))

    for g, start, end in blocks:
        todo = [m for m in models if (m, g) not in done]
        if not todo:
            log.info("block g%d: all done — skip", g)
            continue
        t0 = time.time()
        fit_mask = np.asarray(idx < start)
        block_mask = np.asarray((idx >= start) & (idx < end))
        lab_ok = _label_available_mask(fit_mask, H)
        fit_bars = np.flatnonzero(fit_mask & lab_ok)
        n_val_bars = max(1, int(round(VAL_FRAC * len(fit_bars))))
        tr_bars = fit_bars[:-n_val_bars]
        val_bars = fit_bars[-n_val_bars:]
        blk_bars = np.flatnonzero(block_mask)

        def bar_rows(bars, require_y=True):
            rows = (bars[:, None] * n_assets
                    + np.arange(n_assets)[None, :]).ravel()
            if require_y:
                rows = rows[np.isfinite(y_all[rows])]
            return rows

        tr_rows = bar_rows(tr_bars)
        val_rows = bar_rows(val_bars)
        blk_rows = bar_rows(blk_bars, require_y=False)
        fit_rows = np.concatenate([tr_rows, val_rows])   # tr first, val tail

        log.info("block g%d [%s..%s): assembling X %dx%d (tr %d / val %d)",
                 g, start.date(), end.date(), len(fit_rows) + len(blk_rows),
                 N, len(tr_rows), len(val_rows))
        X_fit = np.empty((len(fit_rows), N), dtype=np.float32)
        X_blk = np.empty((len(blk_rows), N), dtype=np.float32)
        mu_all = np.empty((N, n_assets), dtype=np.float32)
        sd_all = np.empty((N, n_assets), dtype=np.float32)
        fit_idx_rows = fit_mask                          # stats over fit bars
        for j, key in enumerate(skeys):
            sig = pd.read_parquet(SIGNAL_STORE / f"{key}.parquet").reindex(
                index=idx, columns=cols)
            v = np.array(sig.to_numpy(dtype=np.float32), copy=True)
            v[~np.isfinite(v)] = np.nan
            ref = v[fit_idx_rows]
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(ref, axis=0)
                sd = np.nanstd(ref, axis=0, ddof=1)
            sd[sd == 0] = np.nan
            mu_all[j], sd_all[j] = mu, sd
            z = ((v - mu) / sd).ravel()
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            X_fit[:, j] = z[fit_rows]
            X_blk[:, j] = z[blk_rows]
            if (j + 1) % 400 == 0:
                log.info("  %d/%d factors", j + 1, N)
        np.savez_compressed(OUT / "models" / f"zstats_g{g}.npz",
                            mu=mu_all, sd=sd_all, factor_ids=fids)
        del mu_all, sd_all
        y_tr = y_all[tr_rows]
        y_val = y_all[val_rows]
        y_fit = y_all[fit_rows]
        val_tids = tids_all[val_rows]
        fit_tids = tids_all[fit_rows]
        log.info("  assembly done in %.0fs", time.time() - t0)

        def record(model, ic, extra=None, pred_blk=None):
            row = dict(model=model, gen=g, start=str(start.date()),
                       end=str(end.date()), ic=ic, n_factors=N,
                       n_train=int(len(tr_rows)),
                       seconds=round(time.time() - t_model, 1))
            row.update(extra or {})
            with res_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done.add((model, g))
            if pred_blk is not None:
                pd.DataFrame(
                    pred_blk.reshape(len(blk_bars), n_assets),
                    index=idx[blk_bars], columns=cols).to_parquet(
                    OUT / "preds" / f"{model}_g{g}.parquet")
            log.info("  %-8s g%d IC %.5f (%.0fs)", model, g,
                     -9.0 if ic is None else ic, time.time() - t_model)

        # linear baselines on the full fit window
        need_lin = any(m in todo for m in ("equal", "ic", "ridge"))
        if need_lin:
            t_model = time.time()
            fit_ics = np.zeros(N)
            for j in range(N):
                icj = rows_ic(X_fit[:, j].astype(np.float32), y_fit,
                              fit_tids, n_assets)
                fit_ics[j] = 0.0 if icj is None else icj
            np.save(OUT / "models" / f"fit_ics_g{g}.npy", fit_ics)
        for m in ("equal", "ic"):
            if m not in todo:
                continue
            t_model = time.time()
            if m == "equal":
                w = np.sign(fit_ics) / max(1, np.count_nonzero(fit_ics))
            else:
                denom = np.abs(fit_ics).sum() or 1.0
                w = fit_ics / denom
            pred = X_blk.astype(np.float32) @ w.astype(np.float32)
            record(m, block_ic(pred.astype(np.float64), blk_rows, idx, cols,
                               close, block_mask),
                   {"weights": "saved"}, pred)
            np.save(OUT / "models" / f"{m}_w_g{g}.npy", w)
        if "ridge" in todo:
            t_model = time.time()
            G = np.zeros((N, N), dtype=np.float64)
            c = np.zeros(N, dtype=np.float64)
            for a in range(0, len(fit_rows), 16384):
                Xc = X_fit[a:a + 16384].astype(np.float64)
                G += Xc.T @ Xc
                c += Xc.T @ y_fit[a:a + 16384]
            G[np.diag_indices_from(G)] += RIDGE_ALPHA
            w = np.linalg.solve(G, c)
            del G
            pred = X_blk.astype(np.float64) @ w
            record("ridge", block_ic(pred, blk_rows, idx, cols, close,
                                     block_mask),
                   {"alpha": RIDGE_ALPHA}, pred)
            np.save(OUT / "models" / f"ridge_w_g{g}.npy", w)

        if "lasso" in todo:
            # Exact lasso via Gram compression: the objective depends on the
            # data only through (X'X, X'y), so solve the equivalent
            # N x N problem X~ = L', y~ = L^{-1}c with G = LL' (alpha
            # rescaled by n/m).  Alpha chosen on the temporal tail-val by
            # MSE (the trees' convention); model fit on the train rows.
            from sklearn.linear_model import Lasso
            t_model = time.time()
            n_tr_rows = len(tr_rows)
            Xtr = X_fit[: n_tr_rows]
            G = np.zeros((N, N), dtype=np.float64)
            c = np.zeros(N, dtype=np.float64)
            for a in range(0, n_tr_rows, 16384):
                Xc = Xtr[a:a + 16384].astype(np.float64)
                G += Xc.T @ Xc
                c += Xc.T @ y_tr[a:a + 16384]
            G[np.diag_indices_from(G)] += 1e-6 * np.trace(G) / N
            L = np.linalg.cholesky(G)
            y_t = np.linalg.solve(L, c)
            Xt_small = L.T
            y_mean = float(y_tr.mean())
            alpha_max = np.abs(c - G.sum(axis=1) * 0.0).max() / n_tr_rows
            alphas = alpha_max * np.logspace(0, -3.5, 30)
            Xv = X_fit[n_tr_rows:].astype(np.float64)
            # alpha by tail-val pooled IC (the NN convention; val-MSE is
            # too flat at this signal level and can pick the null model)
            best = None
            est = Lasso(alpha=1.0, warm_start=True, fit_intercept=False,
                        max_iter=5000)
            for al in alphas:
                est.alpha = al * n_tr_rows / N
                est.fit(Xt_small, y_t)
                w = est.coef_
                if not (w != 0).any():
                    continue
                vic = rows_ic(Xv @ w, y_val, val_tids, n_assets)
                if vic is not None and (best is None or vic > best[0]):
                    best = (vic, al, w.copy())
            vic, al, w = best
            pred = X_blk.astype(np.float64) @ w
            val_ic = rows_ic(Xv @ w, y_val, val_tids, n_assets)
            np.save(OUT / "models" / f"lasso_w_g{g}.npy", w)
            record("lasso", block_ic(pred, blk_rows, idx, cols, close,
                                     block_mask),
                   {"alpha": float(al), "val_ic": val_ic,
                    "n_nonzero": int((w != 0).sum())}, pred)

        X_tr16 = X_fit[: len(tr_rows)]
        X_val16 = X_fit[len(tr_rows):]
        dtr = dval = None
        if any(m in todo for m in ("lightgbm", "rf")):
            import lightgbm as lgb
            dtr = lgb.Dataset(X_tr16, label=y_tr,
                              params={"max_bin": 127}, free_raw_data=False)
            dval = dtr.create_valid(X_val16, label=y_val)
        for m, mode in (("lightgbm", "gbdt"), ("rf", "rf")):
            if m not in todo:
                continue
            t_model = time.time()
            booster = fit_lightgbm(dtr, dval, val_tids,
                                   n_assets, mode, N, SEED + g)
            pred = predict_chunked_lgb(booster, X_blk)
            val_ic = rows_ic(predict_chunked_lgb(booster, X_val16), y_val,
                             val_tids, n_assets)
            booster.save_model(str(OUT / "models" / f"{m}_g{g}.txt"))
            record(m, block_ic(pred, blk_rows, idx, cols, close, block_mask),
                   {"best_iteration": booster.best_iteration or
                    booster.current_iteration(),
                    "val_ic": val_ic}, pred)
            del booster
        del dtr, dval

        if "nn" in todo:
            import torch
            t_model = time.time()
            model_nn, meta, predict16 = fit_nn(
                X_tr16, y_tr, X_val16, y_val, val_tids, n_assets, SEED + g)
            pred = predict16(X_blk).astype(np.float64)
            torch.save({"state_dict": model_nn.state_dict(), "meta": meta},
                       OUT / "models" / f"nn_g{g}.pt")
            record("nn", block_ic(pred, blk_rows, idx, cols, close,
                                  block_mask),
                   {"val_ic": meta["best_val_ic"],
                    "best_epoch": meta["best_epoch"],
                    "device": meta["device"]}, pred)
            del model_nn
        del X_fit, X_blk, X_tr16, X_val16
        log.info("block g%d complete in %.0f min", g,
                 (time.time() - t0) / 60)

    # summary
    rows = [json.loads(x) for x in res_path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    summ = (df.dropna(subset=["ic"]).groupby("model")["ic"]
            .agg(blockmean="mean", blockstd="std",
                 hit=lambda s: float((s > 0).mean()), n="count"))
    summ.to_csv(OUT / "summary.csv")
    log.info("SUMMARY\n%s", summ.to_string())


if __name__ == "__main__":
    main()
