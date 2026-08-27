"""Point-in-time walk-forward combiner study on the WF panel.

Protocol (user decision 2026-08-05): from 2021-07-20 the panel is split into
the 10 prequential 126-bar (~6-month) blocks the WF runs traded. For EACH
block the combiner is REFIT from scratch on all bars strictly before the
block start (expanding window), using ONLY the factors that existed at that
point in time:

  * evolution arms: the archive snapshot at the END of the generation before
    the block was traded (replay_snapshots on lineage.jsonl — book@gen g
    trades block g+1); SET-mode genomes contribute all member programs;
  * oneshot books / the 101-alpha zoo: the full book from block 1 (their
    research never saw post-2021 data).

The refit composite is scored on the block (pooled per-underlying IC); the
WF statistic is the MEAN of the 10 per-block ICs.

Combination methods (registry below):
  equal      sign(fit-IC)-aligned equal weights on z-scored signals
  ic         weights proportional to fit-window IC
  ridge      catalog RidgeCV (extended alpha grid)
  lasso      catalog LassoCV
  lightgbm   catalog LightGBM defaults
  kakushadze Kakushadze & Yu "How to Combine a Billion Alphas"
             (arXiv:1603.05937) eq. (1): w ~ C^{-1}E on the alphas' own
             daily P&L streams, C from a PC factor model — the paper's own
             prescription for the N << T regime our books live in
  kaku_reg   the paper's verbatim Sec. 5.3 weighted-regression recipe
             (built for N >> T; lookback shrunk to keep that regime, so it
             is honest but expect degradation on small books)
  autoalpha  AutoAlpha (arXiv:2002.08245): top-150 by train IC, PCA-
             similarity diversity gate 0.9, LightGBM(+XGBoost if installed)
             learn-to-rank ensemble with daily query groups

Books: --arm <prerun>|zoo, or a '+'-joined union (e.g. L4WF_terra_s0+zoo);
PIT availability of a union is the union of the members' PIT sets.  A member
may override the global --availability with an '@mode' suffix
(e.g. L4WF_terra_s0@snapshots+L1H_terra_s0b@full+zoo).

Results append to <out-root>/pit_combiners/<label>.jsonl (resume-safe:
(method, block) pairs already present are skipped).
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
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pit_combiners")

H = 6
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
REF_PREQUENTIAL = WS / "L4WF_terra_s0/evolution/prequential.jsonl"
# lightgbm -> rf (user decision 2026-08-06): the remaining PIT runs use the
# catalog random forest as the tree-model combiner; earlier labels keep their
# recorded lightgbm rows (archived by the backfill as *.pre_backfill).
DEFAULT_METHODS = ("equal", "ic", "ridge", "lasso", "rf",
                   "kakushadze", "kaku_reg", "autoalpha")


# ── book / PIT loading ─────────────────────────────────────────────────────

def load_zoo() -> dict[str, str]:
    pb = json.loads((REPO / "data/prebooks/formulaic_101.json").read_text())
    return {m["factor_id"]: m["code"] for m in pb["members"]}


def load_arm(arm: str, availability: str = "snapshots"):
    """(code_by_fid, avail_by_gen | None).

    avail_by_gen maps block-generation (11..20) -> set of factor ids
    available BEFORE that block was traded. None = whole book always
    available (oneshot / zoo).  ``availability="full"`` skips the lineage
    replay and treats the whole kept book as available from block 1 — the
    honest treatment for selection-only arms (L1H: every factor was created
    at generation 0, before any reveal; only archive MEMBERSHIP changes
    later, and the lineage replay does not model the zero-children shape).
    """
    if arm == "zoo":
        return load_zoo(), None

    prerun = WS / arm
    evo_dir = prerun / "evolution"
    db = json.loads((prerun / "factors/factor_db.json").read_text())
    code_by_fid = {}
    for r in db["factors"]:
        path = Path(r["code_path"])
        if not path.exists():
            path = REPO / "quant_fund_agent/factors/researcher" / path.name
        code_by_fid[r["id"]] = path.read_text()

    if not (evo_dir / "lineage.jsonl").exists():
        return code_by_fid, None  # oneshot book

    if availability == "full":
        st = json.loads((evo_dir / "state.json").read_text())
        for eg in st.get("kept_pool", []):
            for prog in eg["genome"]["programs"]:
                code_by_fid.setdefault(prog["factor_id"], prog["code"])
        return code_by_fid, None

    rc = json.loads((evo_dir / "run_config.json").read_text())
    wf_blocks = rc.get("wf_blocks", 0)
    if not wf_blocks:
        raise SystemExit(f"{arm} is not a WF run")
    # run_config["generations"] can be stale (the GP arm records the 10 it was
    # launched with although it ran 20); the lineage is the source of truth.
    gens = rc["generations"]
    with (evo_dir / "lineage.jsonl").open() as f:
        gens = max(gens, max(json.loads(l)["generation"] for l in f))

    if availability == "pool_pit":
        # Whole kept pool, each member revealed one block after the generation
        # that created it: block g (=11..20) sees everything born at gen <= g-1.
        st = json.loads((evo_dir / "state.json").read_text())
        born: dict[str, int] = {}
        for eg in st.get("kept_pool", []) + st.get("archive", []):
            g0 = int(eg["genome"].get("generation") or 0)
            for prog in eg["genome"]["programs"]:
                code_by_fid.setdefault(prog["factor_id"], prog["code"])
                fid = prog["factor_id"]
                born[fid] = min(born.get(fid, g0), g0)
        avail_by_gen = {g: {f for f, b in born.items() if b <= g - 1}
                        for g in range(gens - wf_blocks + 1, gens + 1)}
        return code_by_fid, avail_by_gen

    sys.path.insert(0, str(REPO / "scripts"))
    from prequential_deployment import replay_snapshots

    snap_gens = tuple(range(gens - wf_blocks, gens))  # 10..19
    snapshots = replay_snapshots(evo_dir, snap_gens=snap_gens)

    # genome id -> factor ids (SET genomes carry several)
    fid_by_gid: dict[str, list[str]] = {}
    with (evo_dir / "lineage.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("event") is None:
                fid_by_gid[r["genome_id"]] = list(r.get("factor_ids") or [])

    # kept_pool code beats factor_db (kept_pool is the superset; SET mode
    # materialises every member program)
    st = json.loads((evo_dir / "state.json").read_text())
    for eg in st.get("kept_pool", []):
        for prog in eg["genome"]["programs"]:
            code_by_fid.setdefault(prog["factor_id"], prog["code"])

    avail_by_gen = {}
    for sg, gids in snapshots.items():
        fids = set()
        for gid in gids:
            fids.update(fid_by_gid.get(gid, []))
        avail_by_gen[sg + 1] = fids  # book@gen g trades block g+1
    return code_by_fid, avail_by_gen


# ── combination methods ────────────────────────────────────────────────────
# Every method: fit(X_fit, y_fit, extras) -> predict(X_all) as 1-D array.
# X columns are per-underlying z-scored signals (fit-window stats only).

def _fit_ics(X_fit, y_fit):
    import numpy as np
    ics = np.zeros(X_fit.shape[1])
    ok_y = np.isfinite(y_fit)
    for j in range(X_fit.shape[1]):
        x = X_fit[:, j]
        m = ok_y & np.isfinite(x) & (x != 0)
        if m.sum() > 100:
            with np.errstate(invalid="ignore"):
                c = np.corrcoef(x[m], y_fit[m])[0, 1]
            ics[j] = c if np.isfinite(c) else 0.0
    return ics


def method_equal(X_fit, y_fit, extras):
    import numpy as np
    w = np.sign(_fit_ics(X_fit, y_fit))
    n = max(1, int((w != 0).sum()))
    w = w / n
    return lambda X: X @ w, {"scheme": "sign(ic)/n", "_save": {"weights": w}}


def method_ic(X_fit, y_fit, extras):
    import numpy as np
    ics = _fit_ics(X_fit, y_fit)
    s = np.abs(ics).sum()
    w = ics / s if s > 0 else ics
    return lambda X: X @ w, {"n_nonzero": int((w != 0).sum()),
                             "_save": {"weights": w}}


def _catalog(model):
    from quant_fund_agent.modeling.catalog import build_estimator, _wrap
    if model == "ridge":
        # RidgeCV's stock grid tops out where every WF fit saturated (1e4);
        # extend to 1e6 per the combiner-model study's finding.
        import numpy as np
        from sklearn.linear_model import RidgeCV
        return _wrap(RidgeCV(alphas=np.logspace(-2, 6, 17)))
    if model == "lasso":
        # newer sklearn dropped LassoCV(n_alphas=...) which the catalog uses
        try:
            return build_estimator(model, None)
        except TypeError:
            from sklearn.linear_model import LassoCV
            # precompute=False: the Gram-matrix validation is numerically
            # fragile on float32 feature matrices (sporadic "did not pass
            # validation" failures on 200+-factor books)
            return _wrap(LassoCV(cv=3, random_state=42, precompute=False))
    if model == "rf":
        # catalog random_forest defaults + max_samples=0.25 (each tree
        # bootstraps 25% of rows): sklearn's exact splitter is ~93x slower
        # than LightGBM's histograms at this scale (measured 608s vs 6.6s,
        # 300k x 60, 1 thread), and the bagging fraction buys back ~4x.
        # n_jobs pinned to 2 — sklearn's joblib parallelism ignores
        # OMP_NUM_THREADS and n_jobs=-1 would oversubscribe the shared box.
        from sklearn.ensemble import RandomForestRegressor
        return _wrap(RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=200,
            max_features=0.5, max_samples=0.25, random_state=42, n_jobs=2))
    return build_estimator(model, None)


def method_sklearn(model):
    def fit(X_fit, y_fit, extras):
        import numpy as np
        est = _catalog(model)
        try:
            est.fit(X_fit, y_fit)
        except ValueError as e:
            # LassoCV precomputes its Gram matrix; on a float32 feature matrix
            # sklearn's own validation of that Gram can fail on rounding
            # ("did not pass validation ... we computed X but the user-supplied
            # value was Y").  Retry once in float64 rather than losing the block.
            if "Gram matrix" not in str(e):
                raise
            log.warning("%s: Gram validation failed on float32, refitting "
                        "in float64", model)
            est = _catalog(model)
            est.fit(np.asarray(X_fit, dtype=np.float64), y_fit)
        diag = {"_save": {"model": est}}
        try:
            from quant_fund_agent.modeling.catalog import unwrap_estimator
            inner = unwrap_estimator(est)
            if hasattr(inner, "alpha_"):
                diag["alpha"] = float(inner.alpha_)
            if hasattr(inner, "coef_"):
                c = np.asarray(inner.coef_).ravel()
                diag["n_nonzero_coef"] = int((np.abs(c) > 1e-10).sum())
                diag["_save"]["weights"] = c
        except Exception:  # noqa: BLE001
            pass
        return est.predict, diag
    return fit


def method_kakushadze(X_fit, y_fit, extras):
    """Kakushadze & Yu 'How to Combine a Billion Alphas' (arXiv:1603.05937).

    Their eq. (1): Sharpe-maximising w_i = eta * sum_j C^{-1}_ij E_j on the
    alphas' own daily P&L streams. The paper's Sec. 5.3 regression trick is
    the N >> T limit; for our regime (N factors << T days) their own logic
    says to use eq. (1) with a NONSINGULAR C directly — here a K-PC
    statistical-risk-model covariance (their companion 'Statistical Risk
    Models' construction): C = diag(sd) (V_K L_K V_K' + diag(spec)) diag(sd),
    K = effective rank of the sample correlation matrix, inverted via
    Sherman-Morrison-Woodbury. E_i = mean daily alpha return over the fit
    window; final normalisation sum|w| = 1; composite = sum_i w_i z_i.
    """
    import numpy as np

    R = extras["alpha_daily_returns"]  # T_days x N, fit window only
    ok = np.isfinite(R).all(axis=1)
    R = R[ok]
    T, N = R.shape
    if T < 20:
        raise ValueError("too few days for kakushadze weights")
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1e-12
    Z = (R - mu) / sd
    corr = (Z.T @ Z) / max(1, T - 1)
    eigval, eigvec = np.linalg.eigh(corr)
    eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
    eigval = np.clip(eigval, 0, None)
    # effective rank cap (paper: K based on eRank of the correlation matrix)
    p = eigval / eigval.sum()
    p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    K = max(1, min(int(round(erank)), T - 1, N - 1))
    lam, V = eigval[:K], eigvec[:, :K]
    # factor-model correlation: VK diag(lam) VK' + diag(specific)
    spec = np.clip(1.0 - (V ** 2 * lam).sum(axis=1), 1e-4, None)
    # Sherman-Morrison-Woodbury inverse of (D + V L V')
    Dinv = 1.0 / spec
    B = V * np.sqrt(lam)
    M = np.eye(K) + (B.T * Dinv) @ B
    Minv = np.linalg.inv(M)
    def Cinv(v):
        u = Dinv * v
        return u - Dinv * (B @ (Minv @ (B.T @ u)))
    w = Cinv(mu / sd) / sd  # C = diag(sd) corr diag(sd)
    w[R.std(axis=0) < 1e-10] = 0.0  # dead alpha streams get no weight
    s = np.abs(w).sum()
    if s > 0:
        w = w / s
    return lambda X: X @ w, {"K": K, "erank": round(erank, 2),
                             "n_nonzero": int((w != 0).sum()),
                             "_save": {"weights": w}}


def method_kaku_reg(X_fit, y_fit, extras):
    """Kakushadze & Yu Sec. 5.3 verbatim recipe (weighted regression).

    Built for N alphas >> M lookback days: serially demean the T x N daily
    alpha-return matrix, normalise by sample vols, drop the dependent row,
    cross-sectionally demean the rows (overall-mode removal, their default
    rm.overall=TRUE) and drop one more row; regress E/sigma on the loadings
    (the remaining rows transposed) with no intercept; w = residuals/sigma,
    normalised to sum|w| = 1. Paper default lookback M = 250; because the
    recipe degenerates (residuals -> 0) once the loadings have >= N columns,
    the lookback is shrunk to keep M <= N/2 on small books — recorded in the
    diagnostics so the regime violation is visible, not hidden.
    """
    import numpy as np

    R = extras["alpha_daily_returns"]  # T_days x N, fit window only
    ok = np.isfinite(R).all(axis=1)
    R = R[ok]
    T, N = R.shape
    M = min(250, max(4, N // 2))
    if T < M + 1:
        raise ValueError("too few days for kaku_reg lookback")
    Rw = R[-(M + 1):]                       # most recent M+1 rows
    E = R.mean(axis=0)                      # trailing mean = expected return
    X = Rw - Rw.mean(axis=0)                # step 1: serial demean
    sigma = Rw.std(axis=0, ddof=1)          # step 2: sample vols only
    sigma[sigma < 1e-12] = 1e-12
    Y = (X / sigma)[:-1]                    # steps 3-4: normalise, drop row
    Lam = Y - Y.mean(axis=1, keepdims=True) # step 5: overall-mode removal
    Lam = Lam[:-1]                          # step 6
    L = Lam.T                               # N x (M-1) loadings
    Et = E / sigma                          # step 7
    beta, *_ = np.linalg.lstsq(L, Et, rcond=None)
    eps = Et - L @ beta                     # step 8: residuals
    w = eps / sigma                         # step 9
    w[Rw.std(axis=0) < 1e-10] = 0.0         # dead alpha streams
    s = np.abs(w).sum()
    if s > 1e-12:
        w = w / s                           # step 10
    return lambda Xz: Xz @ w, {"M": M, "N": N,
                               "n_nonzero": int((np.abs(w) > 1e-12).sum()),
                               "regime_ok": bool(N > 2 * M),
                               "_save": {"weights": w}}


def _pc1_stockspace(mat):
    """First principal component (stock space) of a bars x tickers matrix,
    power iteration — AutoAlpha's O(nT + n^2) PCA-similarity proxy."""
    import numpy as np
    A = np.nan_to_num(mat - np.nanmean(mat, axis=1, keepdims=True))
    C = A.T @ A
    v = np.ones(C.shape[0]) / np.sqrt(C.shape[0])
    for _ in range(30):
        v2 = C @ v
        n = np.linalg.norm(v2)
        if n < 1e-12:
            return v
        v2 = v2 / n
        if np.linalg.norm(v2 - v) < 1e-10:
            v = v2
            break
        v = v2
    return v


def method_autoalpha(X_fit, y_fit, extras):
    """AutoAlpha (Zhang et al., IJCAI 2020, arXiv:2002.08245) combiner.

    Faithful to the paper's stated pipeline: (1) per-factor training IC =
    mean daily cross-sectional Pearson corr vs forward returns, sign-flip
    negative-IC factors; (2) greedy diversity filter in IC order using
    PCA-similarity (corr of first stock-space PCs) < 0.9; (3) keep top 150
    by |IC|; (4) LightGBM ranker (lambdarank, daily query groups, 5-bin
    within-day forward-return relevance labels) + XGBoost ranker
    (rank:pairwise) if installed; (5) ensemble = mean of z-scored scores.
    Hyperparameters the paper omits use the defaults recorded here.
    """
    import numpy as np

    X = extras["X"]                    # (n_bars*n_cols) x N, z-scored
    y = extras["y_all"]
    n_cols = extras["n_cols"]
    fit_bars = extras["fit_bar_pos"]   # bar positions inside the fit window
    N = X.shape[1]

    # (1) per-day cross-sectional IC per factor on the fit window
    ic_sum = np.zeros(N)
    ic_cnt = np.zeros(N)
    for b in fit_bars:
        rows = slice(b * n_cols, (b + 1) * n_cols)
        yb = y[rows]
        m = np.isfinite(yb)
        if m.sum() < 10:
            continue
        Xb = X[rows][m].astype(float)
        yb = yb[m]
        yc = yb - yb.mean()
        ys = np.linalg.norm(yc)
        if ys < 1e-12:
            continue
        Xc = Xb - Xb.mean(axis=0)
        xs = np.linalg.norm(Xc, axis=0)
        ok = xs > 1e-12
        ics = np.zeros(N)
        ics[ok] = (Xc[:, ok].T @ yc) / (xs[ok] * ys)
        ic_sum += ics
        ic_cnt += ok.astype(float)
    ic = np.divide(ic_sum, np.maximum(ic_cnt, 1))
    sign = np.where(ic < 0, -1.0, 1.0)

    # (2) greedy PCA-similarity diversity gate (< 0.9), IC-descending
    order = np.argsort(-np.abs(ic))
    pcs = {}

    def pc(j):
        if j not in pcs:
            mat = X[:, j].reshape(-1, n_cols)[fit_bars] * sign[j]
            pcs[j] = _pc1_stockspace(mat)
        return pcs[j]

    kept = []
    for j in order:
        if abs(ic[j]) <= 0:
            continue
        ok = True
        for k in kept:
            c = np.corrcoef(pc(j), pc(k))[0, 1]
            if np.isfinite(c) and abs(c) >= 0.9:
                ok = False
                break
        if ok:
            kept.append(int(j))
        if len(kept) >= 150:               # (3) top-150 cap
            break
    if not kept:
        raise ValueError("autoalpha: no factors survive the diversity gate")
    kept_arr = np.asarray(kept)
    ksign = sign[kept_arr]

    # (4) rank labels: within-day forward-return rank -> 5 relevance bins
    day_of = extras["train_rows"] // n_cols
    ord_rows = np.argsort(day_of, kind="stable")
    rows_sorted = extras["train_rows"][ord_rows]
    days_sorted = day_of[ord_rows]
    Xtr = (X[rows_sorted][:, kept_arr] * ksign).astype(np.float32)
    ytr = y[rows_sorted]
    labels = np.zeros(len(ytr), dtype=int)
    group_sizes = []
    i = 0
    while i < len(days_sorted):
        j = i
        while j < len(days_sorted) and days_sorted[j] == days_sorted[i]:
            j += 1
        r = ytr[i:j].argsort().argsort()
        labels[i:j] = np.minimum((5 * r) // max(1, (j - i)), 4)
        group_sizes.append(j - i)
        i = j

    from lightgbm import LGBMRanker
    rankers = []
    lgb = LGBMRanker(objective="lambdarank", n_estimators=200, num_leaves=31,
                     learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                     random_state=42, n_jobs=-1, verbose=-1)
    lgb.fit(Xtr, labels, group=group_sizes)
    rankers.append(lgb)
    have_xgb = False
    try:
        from xgboost import XGBRanker
        xgb = XGBRanker(objective="rank:pairwise", n_estimators=200,
                        max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb.fit(Xtr, labels, group=group_sizes)
        rankers.append(xgb)
        have_xgb = True
    except ImportError:
        pass

    def predict(Xz):
        parts = []
        Xk = (Xz[:, kept_arr] * ksign).astype(np.float32)
        for r in rankers:
            s = np.asarray(r.predict(Xk), dtype=float)
            sd = s.std()
            parts.append((s - s.mean()) / (sd if sd > 1e-12 else 1.0))
        return np.mean(parts, axis=0)     # (5) score-average ensemble

    return predict, {"n_kept": len(kept), "n_book": N, "xgboost": have_xgb,
                     "top_ic": round(float(np.abs(ic).max()), 4),
                     "_save": {"models": rankers, "kept_idx": kept_arr,
                               "kept_sign": ksign}}


METHODS = {
    "equal": method_equal,
    "ic": method_ic,
    "ridge": method_sklearn("ridge"),
    "lasso": method_sklearn("lasso"),
    "lightgbm": method_sklearn("lightgbm"),
    "rf": method_sklearn("rf"),
    "kakushadze": method_kakushadze,
    "kaku_reg": method_kaku_reg,
    "autoalpha": method_autoalpha,
}


# ── driver ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   help="prerun name, 'zoo', or '+'-joined union")
    p.add_argument("--label", default=None)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    p.add_argument("--out-root",
                   default=str(REPO / "data/comparisons/wf_arm_analysis"))
    p.add_argument("--max-blocks", type=int, default=None,
                   help="smoke-test cap on the number of blocks")
    p.add_argument("--availability",
                   choices=["snapshots", "full", "pool_pit"],
                   default="snapshots",
                   help="'full' = whole kept book available from block 1 "
                        "(selection-only arms whose factors all predate the "
                        "reveals); 'pool_pit' = the whole KEPT POOL, but each "
                        "member only from the block after the generation that "
                        "created it (honest PIT pool race for arms that "
                        "actually evolved across the reveals)")
    p.add_argument("--keep-fids", default=None,
                   help="JSON list of factor ids: restrict the race to these "
                        "(PIT availability is unchanged, just intersected) — "
                        "used to re-race an arm on its non-degenerate subset")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.comparison.standardize import per_underlying_zscore
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from quant_fund_agent.research_eval.harness import (_label_available_mask,
                                                        _pooled_ic)

    label = args.label or args.arm.replace("+", "_plus_").replace("@", "_")
    methods = [m for m in args.methods.split(",") if m]
    out_dir = Path(args.out_root) / "pit_combiners"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"{label}.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["method"], r["block_gen"]))
        log.info("[%s] resume: %d results present", label, len(done))

    # books + PIT availability
    # a member may carry its own availability mode as 'arm@mode' (a union of
    # evolution arms under 'snapshots' and seeding-only arms under 'full');
    # without a suffix the global --availability applies (byte-identical).
    arm_specs = []
    for tok in args.arm.split("+"):
        a, _, mode = tok.partition("@")
        arm_specs.append((a, mode or args.availability))
    arms = [a for a, _ in arm_specs]
    code_by_fid, avail_by_gen = {}, {}
    always_avail = set()
    for a, mode in arm_specs:
        codes, avail = load_arm(a, availability=mode)
        code_by_fid.update(codes)
        if avail is None:
            always_avail.update(codes)
        else:
            for g, fids in avail.items():
                avail_by_gen.setdefault(g, set()).update(fids)

    if args.keep_fids:
        keep = set(json.loads(Path(args.keep_fids).read_text()))
        code_by_fid = {f: c for f, c in code_by_fid.items() if f in keep}
        always_avail &= keep
        avail_by_gen = {g: (f & keep) for g, f in avail_by_gen.items()}
        log.info("[%s] restricted to %d/%d kept factor ids",
                 label, len(code_by_fid), len(keep))

    # blocks (dates from the first evolution member, else the reference)
    src = next((WS / a / "evolution/prequential.jsonl" for a in arms
                if (WS / a / "evolution/prequential.jsonl").exists()),
               REF_PREQUENTIAL)
    blocks = []
    for line in src.read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((r["generation"], pd.Timestamp(r["start"]),
                           pd.Timestamp(r["end"])))
    blocks.sort()
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    n_cols = close.shape[1]
    y_all = forward_returns(close, horizon=H).to_numpy(dtype=float).ravel()
    ret1 = forward_returns(close, horizon=1)

    # signals for every factor that is EVER available
    needed = set(always_avail)
    for fids in avail_by_gen.values():
        needed.update(fids)
    needed &= set(code_by_fid)
    discover_factors()
    sys.path.insert(0, str(REPO / "scripts"))
    from wf_common import load_or_compute_signal
    signals = {}
    for fid in sorted(needed):
        try:
            signals[fid] = load_or_compute_signal(
                fid, code_by_fid[fid], panel, idx, close.columns)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] %s failed: %s", label, fid, e)
    log.info("[%s] %d/%d signals; blocks=%d; methods=%s",
             label, len(signals), len(needed), len(blocks), methods)

    for g, start, end in blocks:
        avail = set(always_avail)
        avail |= avail_by_gen.get(g, set())
        fids = sorted(f for f in avail if f in signals)
        if not fids:
            log.warning("[%s] block g%d: no factors available", label, g)
            continue
        todo = [m for m in methods if (m, g) not in done]
        if not todo:
            continue

        fit_mask = np.asarray(idx < start)
        block_mask = np.asarray((idx >= start) & (idx < end))
        fit_idx = idx[fit_mask]

        # feature matrix (z per underlying, fit-window stats).  Filled column
        # by column into a preallocated float32 buffer: a list of float64
        # ravels + column_stack costs ~3x the memory and OOMs the 8 GB box on
        # big pools (557 factors x 966k rows).  The per-name z frames are only
        # retained when a method actually needs them (the kaku pair).
        need_zsigs = any(m in ("kakushadze", "kaku_reg") for m in todo)
        X = np.empty((len(idx) * n_cols, len(fids)), dtype=np.float32)
        zsigs = {}
        for jj, f in enumerate(fids):
            z = per_underlying_zscore(signals[f].astype(float), fit_idx)
            z = np.nan_to_num(z.to_numpy(dtype=float),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if need_zsigs:
                zsigs[f] = z
            X[:, jj] = z.ravel()
            del z
        lab_ok = _label_available_mask(fit_mask, H)
        fit_rows = np.repeat(np.asarray(fit_mask) & lab_ok, n_cols)
        train = np.flatnonzero(fit_rows & np.isfinite(y_all))
        X_fit, y_fit = X[train], y_all[train]

        extras = {"X": X, "y_all": y_all, "n_cols": n_cols,
                  "fit_bar_pos": np.flatnonzero(np.asarray(fit_mask) & lab_ok),
                  "train_rows": train}
        # daily long-short alpha returns on the fit window (kaku methods)
        if any(m in ("kakushadze", "kaku_reg") for m in todo):
            fit_pos = np.flatnonzero(fit_mask)[:-1]  # last bar has no ret1
            r1 = ret1.to_numpy(dtype=float)
            Rcols = []
            for f in fids:
                zmat = zsigs[f][fit_pos]
                zm = zmat - np.nanmean(zmat, axis=1, keepdims=True)
                gross = np.nansum(np.abs(zm), axis=1)
                gross[gross < 1e-12] = np.nan
                pnl = np.nansum(zm * np.nan_to_num(r1[fit_pos]), axis=1) / gross
                Rcols.append(np.nan_to_num(pnl))  # no positions -> 0 return
            extras["alpha_daily_returns"] = np.column_stack(Rcols)
        del zsigs

        adir = out_dir / "artifacts" / label / f"g{g}"
        for m in todo:
            try:
                predict, diag = METHODS[m](X_fit, y_fit, extras)
                pred = np.asarray(predict(X), dtype=float).reshape(
                    len(idx), n_cols)
                pred_df = pd.DataFrame(pred, index=idx, columns=close.columns)
                ic = _pooled_ic(pred_df, close, H, row_mask=block_mask,
                                available_mask=block_mask)[0]
                ic_fit = _pooled_ic(pred_df, close, H, row_mask=fit_mask,
                                    available_mask=fit_mask)[0]

                # persist the fitted state so nothing needs refitting later:
                # PIT factor list, weight vectors, model dumps, and the OOS
                # block slice of the composite prediction
                save = diag.pop("_save", {}) if isinstance(diag, dict) else {}
                adir.mkdir(parents=True, exist_ok=True)
                fj = adir / "factors.json"
                if not fj.exists():
                    fj.write_text(json.dumps(
                        {"fids": fids, "start": str(start.date()),
                         "end": str(end.date()),
                         "zscore": "per_underlying, fit-window stats"}))
                if save.get("weights") is not None:
                    (adir / f"weights_{m}.json").write_text(json.dumps(
                        {"fids": fids,
                         "weights": [float(x) for x in save["weights"]]}))
                try:
                    import joblib
                    if save.get("model") is not None:
                        joblib.dump(save["model"], adir / f"model_{m}.joblib")
                    if save.get("models") is not None:
                        joblib.dump(
                            {"rankers": save["models"],
                             "kept_fids": [fids[i] for i in save["kept_idx"]],
                             "kept_sign": list(map(float, save["kept_sign"]))},
                            adir / f"model_{m}.joblib")
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] g%d %s model dump failed: %s",
                                label, g, m, e)
                pred_df[block_mask].astype("float32").to_parquet(
                    adir / f"pred_{m}.parquet")

                rec = {"label": label, "method": m, "block_gen": g,
                       "start": str(start.date()), "end": str(end.date()),
                       "n_factors_avail": len(fids), "ic": ic,
                       "ic_fit": ic_fit, "diag": diag,
                       "artifacts": str(adir.relative_to(out_dir))}
                with open(res_path, "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                log.info("[%s] g%d %s: n=%d ic=%.4f", label, g, m,
                         len(fids), ic if ic is not None else float("nan"))
                del pred, pred_df
            except NotImplementedError:
                log.info("[%s] g%d %s: skipped (spec pending)", label, g, m)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] g%d %s FAILED: %s", label, g, m, e)
        del X, X_fit, y_fit, extras

    # summary
    if res_path.exists():
        rows = [json.loads(l) for l in res_path.read_text().splitlines()]
        df = pd.DataFrame(rows)
        summ = (df.dropna(subset=["ic"])
                  .groupby("method")
                  .agg(blockmean=("ic", "mean"), blockstd=("ic", "std"),
                       hit=("ic", lambda s: float((s > 0).mean())),
                       n_blocks=("ic", "size"),
                       mean_n_factors=("n_factors_avail", "mean"))
                  .reset_index())
        summ.insert(0, "label", label)
        summ.to_csv(out_dir / f"{label}_summary.csv", index=False)
        log.info("[%s] summary:\n%s", label, summ.to_string(index=False))


if __name__ == "__main__":
    main()
