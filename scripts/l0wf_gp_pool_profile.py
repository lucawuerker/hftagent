"""Level-class profile of the GP arm's whole KEPT POOL (557 factors).

For every pool member: median per-name lag-1 autocorrelation of the signal
(rho_med, the established level-class statistic — build_clean_pool_prerun.py
uses rho_med < 0.9), the share of names above 0.99, the harness block IC
(in-block Pearson) and the REAL-TIME IC (expanding standardisation +
uncentered cosine, i.e. nothing re-centred on the scored block).

Signals are read straight from the parquet signal store, so no factor is
recomputed and only ``close`` is loaded from the panel.

Writes data/comparisons/l0wf_gp_diagnostics/{pool_profile.csv,
clean_pool_fids.json,clean_book_fids.json}.
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
log = logging.getLogger("poolprof")

import numpy as np
import pandas as pd

WS = REPO / "data/workspaces/fmp_archive_equity_nasdaq100pit/preruns"
ARM = "L0WF_gp_s0"
OUT = REPO / "data/comparisons/l0wf_gp_diagnostics"
H = 6
RHO_MAX = 0.9


def main() -> None:
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.mcp import research_service as svc
    from wf_common import SIGNAL_STORE, signal_key

    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads((WS / ARM / "gp/state.json").read_text())
    book = {e["genome"]["programs"][0]["factor_id"] for e in st["archive"]}
    entries: dict[str, tuple[str, int]] = {}
    for eg in st.get("kept_pool", []) + st.get("archive", []):
        g0 = int(eg["genome"].get("generation") or 0)
        for prog in eg["genome"]["programs"]:
            fid = prog["factor_id"]
            if fid not in entries or g0 < entries[fid][1]:
                entries[fid] = (prog["code"], g0)
    log.info("pool %d factors (%d in book)", len(entries), len(book))

    panel = svc._load_panel_cached("ticker_data", ["close"], n_tickers=None)
    close = panel["close"]
    idx, cols = close.index, close.columns
    y = forward_returns(close, horizon=H).to_numpy(dtype=float)

    blocks = []
    for line in (WS / ARM / "gp/prequential.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("generation", 0) >= 11:
            blocks.append((pd.Timestamp(r["start"]), pd.Timestamp(r["end"])))
    blocks.sort()
    bmasks = [np.asarray((idx >= s) & (idx < e)) for s, e in blocks]

    def ics(x, centered=True):
        out = []
        for m in bmasks:
            xb, yb = x[m], y[m]
            num = den = 0.0
            for j in range(xb.shape[1]):
                a, b = xb[:, j], yb[:, j]
                ok = np.isfinite(a) & np.isfinite(b)
                n = int(ok.sum())
                if n < 10:
                    continue
                a, b = a[ok], b[ok]
                if centered:
                    a, b = a - a.mean(), b - b.mean()
                sa, sb = np.linalg.norm(a), np.linalg.norm(b)
                if sa < 1e-12 or sb < 1e-12:
                    continue
                num += n * float(a @ b / (sa * sb))
                den += n
            out.append(num / den if den else np.nan)
        return out

    rows, missing = [], 0
    for k, (fid, (code, gen)) in enumerate(sorted(entries.items()), 1):
        p = SIGNAL_STORE / f"{signal_key(fid, code)}.parquet"
        if not p.exists():
            missing += 1
            continue
        sig = pd.read_parquet(p).reindex(index=idx, columns=cols)
        x = sig.to_numpy(dtype=float)
        rhos = []
        for j in range(x.shape[1]):
            a = x[:, j][np.isfinite(x[:, j])]
            if len(a) < 100 or a.std() < 1e-12:
                continue
            c = np.corrcoef(a[:-1], a[1:])[0, 1]
            if np.isfinite(c):
                rhos.append(abs(c))
        if not rhos:
            continue
        rho_med = float(np.median(rhos))
        # real-time: expanding standardisation, uncentered scoring
        mu = sig.expanding(min_periods=252).mean()
        sd = sig.expanding(min_periods=252).std()
        z = (sig - mu).div(sd.where(sd > 1e-12)).to_numpy(dtype=float)
        ic_raw, ic_rt = ics(x), ics(z, centered=False)
        rows.append({
            "factor_id": fid, "generation": gen, "in_book": fid in book,
            "rho_med": rho_med,
            "share_rho_gt_099": float(np.mean(np.array(rhos) > 0.99)),
            "ic_raw_mean": float(np.nanmean(ic_raw)),
            "ic_raw_absmean": float(np.nanmean(np.abs(ic_raw))),
            "ic_realtime_mean": float(np.nanmean(ic_rt)),
            "ic_realtime_absmean": float(np.nanmean(np.abs(ic_rt))),
        })
        if k % 50 == 0:
            log.info("%d/%d", k, len(entries))

    df = pd.DataFrame(rows).sort_values("rho_med")
    df.to_csv(OUT / "pool_profile.csv", index=False)
    clean = df[df.rho_med < RHO_MAX]
    (OUT / "clean_pool_fids.json").write_text(
        json.dumps(sorted(clean.factor_id), indent=1))
    (OUT / "clean_book_fids.json").write_text(
        json.dumps(sorted(clean[clean.in_book].factor_id), indent=1))
    log.info("profiled %d (missing signals %d); clean (rho<%.2f): %d pool / "
             "%d book; mean|ic_raw| %.4f vs |ic_rt| %.4f (all) and %.4f/%.4f "
             "(clean)", len(df), missing, RHO_MAX, len(clean),
             int(clean.in_book.sum()), df.ic_raw_absmean.mean(),
             df.ic_realtime_absmean.mean(), clean.ic_raw_absmean.mean(),
             clean.ic_realtime_absmean.mean())


if __name__ == "__main__":
    main()
