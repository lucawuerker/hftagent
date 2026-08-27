"""Arm-4 (L1H_terra_s0) deep-dive for the thesis ablation chapter.

Curated published book (factor_db, 22 members) vs the full kept pool
(state.json kept_pool, 80 members):

  * pooled signal correlation matrices for both (CSV + clustered heatmaps),
    mean |rho|, max |rho|, effective N for both;
  * per-factor fit-window IC and 10-block WF ICs for EVERY pool member
    (the arm-analysis suite only covers the published book);
  * mechanism-group structure of pool vs book (+ factor categories);
  * Pareto-axis values of the final archive members;
  * PIT-Lasso selection frequency per factor across the 10 blocks, for the
    pool race and (if present) the curated-book race.

Outputs: data/comparisons/thesis_ablation/l1h_deepdive/{tables,figures}/.
Signal computation goes through the shared parquet signal store, so re-runs
are cheap.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ARM = "L1H_terra_s0"
H = 6
WF_START = "2021-07-20"
WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns" / ARM
OUT = REPO / "data/comparisons/thesis_ablation/l1h_deepdive"
TAB, FIG = OUT / "tables", OUT / "figures"
PIT_ART = REPO / "data/comparisons/wf_arm_analysis_local/pit_combiners/artifacts"

C_BOOK = "#2a78d6"
C_POOL = "#eb6834"
C_GRID = "#e4e3df"
C_TEXT = "#0b0b0b"
C_MUT = "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "thesis_div", ["#2a78d6", "#f2f1ee", "#e34948"])


def style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUT, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def load_members():
    db = json.loads((WS / "factors/factor_db.json").read_text())
    book_ids = [r["id"] for r in db["factors"]]
    meta = {r["id"]: dict(category=r.get("category"),
                          horizon=r.get("prediction_horizon"))
            for r in db["factors"]}
    st = json.loads((WS / "evolution/state.json").read_text())
    pool = {}
    for eg in st["kept_pool"]:
        g = eg["genome"]
        for prog in g["programs"]:
            pool[prog["factor_id"]] = dict(
                code=prog["code"], group=g.get("mechanism_group_id"),
                mechanism=(prog.get("metadata") or {}).get("mechanism")
                or (g.get("metadata") or {}).get("mechanism"))
    # archive fitness (final Pareto members)
    arch = []
    for a in st["archive"]:
        g = a["genome"]
        fid = g["programs"][0]["factor_id"]
        obj = a["fitness"].get("objective") or {}
        arch.append(dict(factor_id=fid, group=g.get("mechanism_group_id"),
                         **{k: obj.get(k) for k in
                            ("marginal_value", "independence", "parsimony",
                             "structural_novelty")}))
    groups = json.loads((WS / "evolution/mechanism_groups.json").read_text())
    return book_ids, meta, pool, pd.DataFrame(arch), groups


def compute_signals(pool):
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc
    from wf_common import load_or_compute_signal

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    signals = {}
    for fid, m in pool.items():
        try:
            signals[fid] = load_or_compute_signal(fid, m["code"], panel, idx,
                                                  close.columns)
        except Exception as e:  # noqa: BLE001
            print("signal failed:", fid, e)
    return panel, close, idx, signals


def corr_stats(mat_ids, signals, fit_pos, stride_target=400):
    stride = max(1, len(fit_pos) // stride_target)
    rows_sel = fit_pos[::stride]
    cols, ids, degenerate = [], [], []
    for fid in mat_ids:
        if fid not in signals:
            continue
        v = np.nan_to_num(
            signals[fid].astype(float).to_numpy()[rows_sel].ravel())
        if np.std(v) < 1e-12:
            degenerate.append(fid)
            continue
        cols.append(v)
        ids.append(fid)
    c = np.corrcoef(np.column_stack(cols), rowvar=False)
    n = c.shape[0]
    off = c[np.triu_indices(n, 1)]
    off = off[np.isfinite(off)]
    eig = np.clip(np.linalg.eigvalsh(np.nan_to_num(c)), 0, None)
    stats = dict(n=n, n_degenerate=len(degenerate),
                 mean_abs_corr=float(np.mean(np.abs(off))),
                 max_abs_corr=float(np.max(np.abs(off))),
                 n_eff=float(eig.sum() ** 2 / (eig ** 2).sum()))
    return pd.DataFrame(c, index=ids, columns=ids), stats


def cluster_order(c: pd.DataFrame) -> list[str]:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    d = 1 - np.abs(np.nan_to_num(c.to_numpy()))
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2
    z = linkage(squareform(d, checks=False), method="average")
    return [c.index[i] for i in leaves_list(z)]


def heatmap(c: pd.DataFrame, stats: dict, title: str, name: str,
            label_fs: float):
    order = cluster_order(c)
    c = c.loc[order, order]
    n = len(c)
    size = min(13.5, max(6.0, 0.42 * n + 2.2))
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(c.to_numpy(), cmap=DIVERGING,
                   norm=TwoSlopeNorm(0., -1, 1))
    short = [i[:34] for i in c.index]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=label_fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=label_fs)
    ax.tick_params(colors=C_MUT, length=0)
    deg = (f"   ({stats['n_degenerate']} degenerate excluded)"
           if stats.get("n_degenerate") else "")
    ax.set_title(
        f"{title}\n$n$={stats['n']}   mean$|\\rho|$="
        f"{stats['mean_abs_corr']:.3f}   max$|\\rho|$="
        f"{stats['max_abs_corr']:.2f}   $N_{{eff}}$={stats['n_eff']:.1f}"
        + deg, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.7, label="pooled signal correlation")
    save(fig, name)


def per_factor_blocks(pool, book_ids, close, idx, signals):
    from quant_fund_agent.research_eval.harness import _pooled_ic

    fit_mask = np.asarray(idx < pd.Timestamp(WF_START))
    blocks = []
    for line in (WS / "evolution/prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            m = np.asarray((idx >= pd.Timestamp(r["start"]))
                           & (idx < pd.Timestamp(r["end"])))
            blocks.append((r["generation"], m))
    rows = []
    for fid, sig in signals.items():
        sigf = sig.astype(float)
        rec = dict(factor_id=fid, in_book=fid in book_ids,
                   group=pool[fid]["group"],
                   ic_fit=_pooled_ic(sigf, close, H, row_mask=fit_mask,
                                     available_mask=fit_mask)[0])
        bl = []
        for g, m in blocks:
            ic = _pooled_ic(sigf, close, H, row_mask=m, available_mask=m)[0]
            rec[f"ic_g{g}"] = ic
            if ic is not None:
                bl.append(ic)
        rec["ic_wf_blockmean"] = float(np.mean(bl)) if bl else None
        rec["wf_hit"] = float(np.mean([i > 0 for i in bl])) if bl else None
        rows.append(rec)
    return pd.DataFrame(rows)


def lasso_usage(label: str) -> pd.DataFrame | None:
    import joblib
    root = PIT_ART / label
    if not root.exists():
        return None
    rows = []
    for g in range(11, 21):
        d = root / f"g{g}"
        mp, fp = d / "model_lasso.joblib", d / "factors.json"
        if not (mp.exists() and fp.exists()):
            continue
        fids = json.loads(fp.read_text())
        if isinstance(fids, dict):
            fids = (fids.get("fids") or fids.get("factor_ids")
                    or fids.get("factors") or [])
        est = joblib.load(mp)
        for attr in ("best_estimator_", "regressor_"):
            est = getattr(est, attr, est)
        if hasattr(est, "steps"):
            est = est.steps[-1][1]
        coef = getattr(est, "coef_", None)
        if coef is None:
            continue
        for fid, c in zip(fids, np.ravel(coef)):
            rows.append(dict(block=g - 10, factor_id=fid, coef=float(c),
                             selected=bool(abs(c) > 1e-12)))
    return pd.DataFrame(rows) if rows else None


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    book_ids, meta, pool, arch, groups = load_members()
    print(f"book {len(book_ids)}, pool {len(pool)}, archive {len(arch)}")
    panel, close, idx, signals = compute_signals(pool)
    fit_pos = np.flatnonzero(np.asarray(idx < pd.Timestamp(WF_START)))

    # 1. correlation matrices, book vs pool -------------------------------
    c_book, s_book = corr_stats(book_ids, signals, fit_pos)
    c_pool, s_pool = corr_stats(list(pool), signals, fit_pos)
    c_book.to_csv(TAB / "corr_book.csv")
    c_pool.to_csv(TAB / "corr_pool.csv")
    pd.DataFrame([dict(set="curated book", **s_book),
                  dict(set="kept pool", **s_pool)]).to_csv(
        TAB / "independence_book_vs_pool.csv", index=False)
    heatmap(c_book, s_book, "L1H curated book — signal correlation",
            "a1_corr_book.png", 7.0)
    heatmap(c_pool, s_pool, "L1H full kept pool — signal correlation",
            "a2_corr_pool.png", 4.6)

    # 2. per-factor fit vs WF ICs, whole pool ------------------------------
    pf = per_factor_blocks(pool, set(book_ids), close, idx, signals)
    pf.to_csv(TAB / "pool_per_factor_blocks.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for sel, col, lab, s in ((False, C_POOL, "pool only", 26),
                             (True, C_BOOK, "curated book", 46)):
        d = pf[pf["in_book"] == sel]
        ax.scatter(d["ic_fit"], d["ic_wf_blockmean"], s=s, color=col,
                   alpha=0.75, edgecolor="white", lw=0.8, label=lab,
                   zorder=3 if sel else 2)
    lim = max(pf["ic_fit"].abs().max(), pf["ic_wf_blockmean"].abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=C_GRID, lw=1, zorder=1)
    ax.axhline(0, color=C_MUT, lw=0.8)
    ax.axvline(0, color=C_MUT, lw=0.8)
    ax.set_xlabel("pooled IC on the fit window (pre-2021-07)", fontsize=10)
    ax.set_ylabel("mean WF block IC (2021-07 – 2026-07)", fontsize=10)
    style_ax(ax)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=9)
    r_all = np.corrcoef(pf["ic_fit"].fillna(0), pf["ic_wf_blockmean"]
                        .fillna(0))[0, 1]
    flips_b = pf[pf["in_book"]]
    flips_p = pf
    phi_b = (np.sign(flips_b["ic_fit"]) != np.sign(
        flips_b["ic_wf_blockmean"])).mean()
    phi_p = (np.sign(flips_p["ic_fit"]) != np.sign(
        flips_p["ic_wf_blockmean"])).mean()
    ax.set_title(f"fit-window vs walk-forward IC   corr {r_all:.2f}   "
                 f"$\\Phi$ book {phi_b:.0%} / pool {phi_p:.0%}", fontsize=10)
    save(fig, "a3_fit_vs_wf_scatter.png")

    # 3. per-book-factor block heatmap ------------------------------------
    d = pf[pf["in_book"]].sort_values("ic_wf_blockmean", ascending=False)
    mat = d[[f"ic_g{g}" for g in range(11, 21)]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(d) + 1.6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap=DIVERGING, norm=TwoSlopeNorm(0., -vmax, vmax),
                   aspect="auto")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f[:36] for f in d["factor_id"]], fontsize=7)
    ax.set_xticks(range(10))
    ax.set_xticklabels(range(1, 11), fontsize=8)
    ax.set_xlabel("walk-forward block", fontsize=10)
    ax.tick_params(colors=C_MUT, length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, label="block IC")
    ax.set_title("L1H curated book — per-factor walk-forward block ICs",
                 fontsize=11)
    save(fig, "a4_book_factor_blocks.png")

    # 4. mechanism-group structure ----------------------------------------
    def group_label(g):
        gid = g["mechanism_group_id"]
        if g.get("mechanisms"):
            return f"G{gid}: {g['mechanisms'][0]}"
        focus = str(g.get("focus") or "").replace("mechanism(s)", "mechanisms")
        if "(" in focus:
            first = focus.split("(", 1)[1].split(",")[0].split(")")[0]
            return f"G{gid}: {first.strip()}"
        return f"group {gid}"

    gsl = {g["mechanism_group_id"]: group_label(g) for g in groups}
    pool_g = pd.Series({fid: m["group"] for fid, m in pool.items()})
    book_g = pool_g[pool_g.index.isin(book_ids)]
    cnt = pd.DataFrame({
        "pool": pool_g.value_counts().sort_index(),
        "book": book_g.value_counts().sort_index()}).fillna(0)
    cnt.index = [str(gsl.get(g, g))[:40] for g in cnt.index]
    cnt.to_csv(TAB / "group_structure.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(cnt))
    ax.barh(y + 0.19, cnt["pool"], height=0.38, color=C_POOL,
            label="kept pool")
    ax.barh(y - 0.19, cnt["book"], height=0.38, color=C_BOOK,
            label="curated book")
    for yi, (p, b) in zip(y, zip(cnt["pool"], cnt["book"])):
        ax.text(p + 0.15, yi + 0.19, f"{p:.0f}", va="center", fontsize=8,
                color=C_TEXT)
        ax.text(b + 0.15, yi - 0.19, f"{b:.0f}", va="center", fontsize=8,
                color=C_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(cnt.index, fontsize=8)
    ax.set_xlabel("factors", fontsize=10)
    ax.invert_yaxis()
    style_ax(ax)
    ax.xaxis.grid(True, color=C_GRID, lw=0.8)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("L1H — factors per mechanism group", fontsize=11)
    save(fig, "a5_group_structure.png")

    # 5. Pareto axes of the archive ---------------------------------------
    arch.to_csv(TAB / "archive_pareto_axes.csv", index=False)
    axes_cols = ["marginal_value", "independence", "parsimony",
                 "structural_novelty"]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.0))
    for ax, col in zip(axs, axes_cols):
        v = arch[col].dropna()
        ax.scatter(np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v,
                   s=26, color=C_BOOK, alpha=0.75, edgecolor="white", lw=0.7)
        ax.hlines(v.median(), -0.2, 0.2, color=C_TEXT, lw=1.4)
        ax.set_xlim(-0.35, 0.35)
        ax.set_xticks([])
        ax.set_title(col.replace("_", " "), fontsize=9.5)
        style_ax(ax)
    fig.suptitle("L1H final archive — objective-vector axes "
                 "(median marked)", fontsize=11, y=1.04)
    save(fig, "a6_pareto_axes.png")

    # 6. PIT-Lasso selection frequency ------------------------------------
    for label, kind in ((ARM, "pool"), ("L1HCUR_terra_s0", "book")):
        u = lasso_usage(label)
        if u is None:
            print("no lasso artifacts for", label)
            continue
        u.to_csv(TAB / f"lasso_usage_{kind}.csv", index=False)
        freq = (u.groupby("factor_id")["selected"].mean()
                .sort_values(ascending=False))
        sel = freq[freq > 0]
        fig, ax = plt.subplots(figsize=(7.0, 0.24 * len(sel) + 1.4))
        yy = np.arange(len(sel))
        ax.barh(yy, sel.to_numpy(), height=0.62,
                color=C_BOOK if kind == "book" else C_POOL)
        ax.set_yticks(yy)
        ax.set_yticklabels([f[:38] for f in sel.index], fontsize=7)
        ax.set_xlabel("share of the 10 blocks with nonzero Lasso weight",
                      fontsize=10)
        ax.invert_yaxis()
        style_ax(ax)
        ax.xaxis.grid(True, color=C_GRID, lw=0.8)
        ax.yaxis.grid(False)
        never = int((freq == 0).sum())
        ax.set_title(f"L1H {kind} — PIT-Lasso selection frequency "
                     f"({never} of {len(freq)} never selected)", fontsize=10)
        save(fig, f"a7_lasso_usage_{kind}.png")

    # 7. categories of the book -------------------------------------------
    cats = pd.Series({fid: meta[fid]["category"] for fid in book_ids})
    cats.value_counts().to_csv(TAB / "book_categories.csv")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
