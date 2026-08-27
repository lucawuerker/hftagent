"""Diagnostics for the GP benchmark arm (L0WF_gp_s0): is its record real?

Three independent checks, all on the WF panel and the same 10 prequential
126-bar blocks the arm traded (2021-07-20 -> 2026-07-27):

1. CAUSALITY   recompute every factor on a panel truncated at 2023-01-02 and
   compare with the full-panel signal on the overlap.  Any disagreement means
   calc() reads the future (the literal look-ahead found in the 4o-mini arm).
   The GP grammar is causal by construction, so this is expected to be clean —
   it separates "leakage" from "metric gaming".

2. LEVEL CLASS median per-name lag-1 autocorrelation of the signal (rho_med)
   and the share of names above 0.99.  rho_med >= 0.9 is the established
   level-class threshold (build_clean_pool_prerun.py).

3. METRIC ARTIFACT  the per-underlying pooled IC (_pooled_ic) centres the
   signal over the WHOLE scored block, so a non-stationary level is implicitly
   demeaned against a window mean that contains the future.  For each factor we
   compare
     ic_raw     the harness statistic (block-window centring)
     ic_causal  the same signal standardised per name with EXPANDING stats
                (only past bars) before scoring — what a deployment sees
   plus reference series: log(close) (pure price level, zero real forecasting
   content), an independent random walk, and a stationary control.

Writes data/comparisons/l0wf_gp_diagnostics/{per_factor.csv,controls.csv,
REPORT.md}.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("gpdiag")

import numpy as np
import pandas as pd

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
ARM = "L0WF_gp_s0"
OUT = REPO / "data/comparisons/l0wf_gp_diagnostics"
H = 6
TRUNC = pd.Timestamp("2023-01-02")


def main() -> None:
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc
    from wf_common import load_or_compute_signal

    OUT.mkdir(parents=True, exist_ok=True)
    discover_factors()

    st = json.loads((WS / ARM / "gp/state.json").read_text())
    book = {}          # fid -> (code, description, generation)
    for eg in st["archive"]:
        pr = eg["genome"]["programs"][0]
        book[pr["factor_id"]] = (pr["code"], pr.get("description", ""),
                                 eg["genome"].get("generation"))
    log.info("book: %d factors", len(book))

    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx, cols = close.index, close.columns
    y = np.array(forward_returns(close, horizon=H).to_numpy(dtype=float),
                 copy=True)

    blocks = []
    for line in (WS / ARM / "gp/prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((r["generation"], pd.Timestamp(r["start"]),
                           pd.Timestamp(r["end"])))
    blocks.sort()
    bmasks = [(g, np.asarray((idx >= s) & (idx < e))) for g, s, e in blocks]
    log.info("blocks: %d (%s -> %s)", len(blocks), blocks[0][1].date(),
             blocks[-1][2].date())

    # ── scoring helpers ────────────────────────────────────────────────────
    def block_ics(x: np.ndarray, target: np.ndarray | None = None) -> list[float]:
        """Per-underlying pooled Pearson IC per block — the harness statistic."""
        tgt = y if target is None else target
        out = []
        for _, m in bmasks:
            xb, yb = x[m], tgt[m]
            num = den = 0.0
            for j in range(xb.shape[1]):
                a, b = xb[:, j], yb[:, j]
                ok = np.isfinite(a) & np.isfinite(b)
                n = int(ok.sum())
                if n < 10:
                    continue
                ac, bc = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
                sa, sb = np.linalg.norm(ac), np.linalg.norm(bc)
                if sa < 1e-12 or sb < 1e-12:
                    continue
                num += n * float(ac @ bc / (sa * sb))
                den += n
            out.append(num / den if den else np.nan)
        return out

    def block_ics_rt(x: np.ndarray, min_obs: int = 252,
                     target: np.ndarray | None = None) -> list[float]:
        """Real-time IC: signal standardised with EXPANDING (past-only) stats
        and scored with an UNCENTERED cosine, so nothing in the statistic is
        re-centred on the scored block.  This is the number a deployment can
        actually earn."""
        df = pd.DataFrame(x, index=idx, columns=cols)
        mu = df.expanding(min_periods=min_obs).mean()
        sd = df.expanding(min_periods=min_obs).std()
        z = (df - mu).div(sd.where(sd > 1e-12)).to_numpy(dtype=float)
        tgt = y if target is None else target
        out = []
        for _, m in bmasks:
            xb, yb = z[m], tgt[m]
            num = den = 0.0
            for j in range(xb.shape[1]):
                a, b = xb[:, j], yb[:, j]
                ok = np.isfinite(a) & np.isfinite(b)
                n = int(ok.sum())
                if n < 10:
                    continue
                a, b = a[ok], b[ok]
                sa, sb = np.linalg.norm(a), np.linalg.norm(b)
                if sa < 1e-12 or sb < 1e-12:
                    continue
                num += n * float(a @ b / (sa * sb))
                den += n
            out.append(num / den if den else np.nan)
        return out

    def causal_z(x: np.ndarray, min_obs: int = 252) -> np.ndarray:
        """Per-name EXPANDING z-score: only bars <= t enter the stats."""
        df = pd.DataFrame(x, index=idx, columns=cols)
        mu = df.expanding(min_periods=min_obs).mean()
        sd = df.expanding(min_periods=min_obs).std()
        return (df - mu).div(sd.where(sd > 1e-12)).to_numpy(dtype=float)

    def level_rho(x: np.ndarray) -> tuple[float, float]:
        rhos = []
        for j in range(x.shape[1]):
            a = x[:, j]
            ok = np.isfinite(a)
            if ok.sum() < 100:
                continue
            a = a[ok]
            if a.std() < 1e-12:
                continue
            c = np.corrcoef(a[:-1], a[1:])[0, 1]
            if np.isfinite(c):
                rhos.append(abs(c))
        if not rhos:
            return np.nan, np.nan
        return float(np.median(rhos)), float(np.mean(np.array(rhos) > 0.99))

    # ── 1. causality: truncated-panel recompute ────────────────────────────
    tpanel = {k: (v.loc[v.index < TRUNC] if isinstance(v, pd.DataFrame) else v)
              for k, v in panel.items()}
    tidx = close.index[close.index < TRUNC]

    rows = []
    for fid, (code, desc, gen) in sorted(book.items()):
        sig = load_or_compute_signal(fid, code, panel, idx, cols)
        x = sig.to_numpy(dtype=float)

        # Causality: FULL vs TRUNCATED panel, both recomputed in this process.
        # (Comparing against the parquet signal store instead would compare two
        # panel VINTAGES — the store's signals were computed on an earlier one —
        # and flag every factor.  That drift is reported separately below.)
        try:
            cls = compile_factor(code, fid)
            full = compute_signal(cls, panel).reindex(
                index=idx, columns=cols).to_numpy(dtype=float)
            tsig = compute_signal(cls, tpanel).reindex(index=tidx, columns=cols)
            a = full[:len(tidx)]
            b = tsig.to_numpy(dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            diff = both & ~np.isclose(a, b, rtol=1e-6, atol=1e-9)
            n_diff = int(diff.sum())
            causal_ok = n_diff == 0
            first_bad = (str(tidx[np.flatnonzero(diff.any(axis=1))[0]].date())
                         if n_diff else "")
            # also: does the stored signal match a fresh compute? (vintage drift)
            bothv = np.isfinite(full) & np.isfinite(x)
            vd = bothv & ~np.isclose(full, x, rtol=1e-4, atol=1e-8)
            n_vintage = int(vd.sum())
            vintage_share = float(n_vintage / max(int(bothv.sum()), 1))
        except Exception as e:  # noqa: BLE001
            causal_ok, n_diff, first_bad = None, -1, f"ERR {e}"[:60]
            n_vintage, vintage_share = -1, float("nan")

        rho, share99 = level_rho(x)
        ic_raw = block_ics(x)
        ic_cau = block_ics(causal_z(x))
        ic_rt = block_ics_rt(x)
        rows.append({
            "factor_id": fid, "generation": gen, "expr": desc[:160],
            "causal_ok": causal_ok, "n_disagree": n_diff,
            "first_disagreement": first_bad,
            "store_vs_fresh_disagree_share": vintage_share,
            "rho_med": rho, "share_rho_gt_099": share99,
            "ic_raw_mean": float(np.nanmean(ic_raw)),
            "ic_raw_absmean": float(np.nanmean(np.abs(ic_raw))),
            "ic_raw_hit": float(np.mean(np.sign(ic_raw) == np.sign(np.nanmean(ic_raw)))),
            "ic_causal_mean": float(np.nanmean(ic_cau)),
            "ic_causal_absmean": float(np.nanmean(np.abs(ic_cau))),
            "ic_realtime_mean": float(np.nanmean(ic_rt)),
            "ic_realtime_absmean": float(np.nanmean(np.abs(ic_rt))),
            "retention_abs": float(np.nanmean(np.abs(ic_cau))
                                   / max(np.nanmean(np.abs(ic_raw)), 1e-12)),
            **{f"b{g}": v for (g, _), v in zip(bmasks, ic_raw)},
        })
        log.info("%-10s rho=%.3f ic_raw=%+.4f ic_causal=%+.4f causal_ok=%s",
                 fid, rho, rows[-1]["ic_raw_mean"], rows[-1]["ic_causal_mean"],
                 causal_ok)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_factor.csv", index=False)

    # ── 3. reference series ────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    logc = np.log(close.to_numpy(dtype=float))
    ret = np.diff(logc, axis=0, prepend=np.nan)
    finite = np.isfinite(close.to_numpy(dtype=float))

    rw = np.cumsum(np.where(np.isfinite(ret), 0.0, 0.0)
                   + rng.normal(0, 0.02, size=logc.shape), axis=0)
    rw = np.where(finite, rw, np.nan)

    ctrls = {
        "log(close)  [pure price level]": (logc, None),
        "close       [pure price level]": (close.to_numpy(dtype=float), None),
        "indep. random walk vs REAL returns": (rw, None),
        "returns     [stationary control]": (ret, None),
    }
    # The decisive control: a PURELY SYNTHETIC random walk scored against ITS
    # OWN 6-bar forward increment.  No market data enters at all, so any IC it
    # earns is a property of the statistic, not of the data.
    rw_fwd = np.full_like(rw, np.nan)
    rw_fwd[:-H] = rw[H:] - rw[:-H]
    ctrls["synthetic RW vs its OWN fwd increment"] = (rw, rw_fwd)

    crows = []
    for name, (x, yy) in ctrls.items():
        rho, s99 = level_rho(x)
        if yy is None:
            ic_raw, ic_cau, ic_rt = (block_ics(x), block_ics(causal_z(x)),
                                     block_ics_rt(x))
        else:
            ic_raw = block_ics(x, target=yy)
            ic_cau = block_ics(causal_z(x), target=yy)
            ic_rt = block_ics_rt(x, target=yy)
        crows.append({"series": name, "rho_med": rho, "share_rho_gt_099": s99,
                      "ic_raw_mean": float(np.nanmean(ic_raw)),
                      "ic_raw_absmean": float(np.nanmean(np.abs(ic_raw))),
                      "ic_causal_mean": float(np.nanmean(ic_cau)),
                      "ic_realtime_mean": float(np.nanmean(ic_rt)),
                      "ic_realtime_absmean": float(np.nanmean(np.abs(ic_rt)))})
        log.info("CTRL %-38s rho=%.3f ic_raw=%+.4f ic_rt=%+.4f",
                 name, rho, crows[-1]["ic_raw_mean"],
                 crows[-1]["ic_realtime_mean"])
    cdf = pd.DataFrame(crows)
    cdf.to_csv(OUT / "controls.csv", index=False)

    clean = df[(df.rho_med < 0.9) & (df.causal_ok != False)]  # noqa: E712
    (OUT / "clean_factor_ids.json").write_text(
        json.dumps(sorted(clean.factor_id), indent=1))

    lines = [f"# {ARM} — GP benchmark diagnostics", "",
             f"Book: {len(df)} factors. Blocks: {len(blocks)} "
             f"({blocks[0][1].date()} -> {blocks[-1][2].date()}), horizon {H}.",
             "", "## Per factor", "",
             df[["factor_id", "generation", "rho_med", "share_rho_gt_099",
                 "causal_ok", "n_disagree", "ic_raw_mean", "ic_causal_mean",
                 "ic_realtime_mean", "expr"]].to_string(index=False),
             "", "## Reference series (same metric, same blocks)", "",
             cdf.to_string(index=False), "",
             f"## Level-clean subset (rho_med < 0.9, causal): "
             f"{len(clean)}/{len(df)}", "",
             ", ".join(sorted(clean.factor_id)) or "(none)", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    log.info("DONE -> %s", OUT)


if __name__ == "__main__":
    main()
