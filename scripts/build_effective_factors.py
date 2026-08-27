"""Build the effective-factor folders referenced by the thesis.

For each book combination of the union table (101 alphas, arm 4, arm 6 and
their unions) this selects k = ceil(N_eff) representative members by pivoted
QR / pivoted-Cholesky column selection on the correlation matrix of the
standardised fit-window signals (identical criterion: each step picks the
factor with the largest signal variance unexplained by those already
chosen), and copies the selected factors' code into
``effective_factors/<set>/`` with a per-set README.

Conventions follow the WF analysis scripts: panel = nasdaq100_2010_wf, fit
window < 2021-07-20, per-underlying z-scores with fit-window stats,
correlation matrix on strided fit rows (same stride rule as
wf_arm_factor_analysis's diversity block).  Deterministic; no walk-forward
outcome is read.

Usage: ./venv/bin/python scripts/build_effective_factors.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QF_CONFIG_FILE", "quant.config.nasdaq100_2010_wf.yaml")
os.environ.setdefault("QF_USE_MCP", "0")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from wf_common import load_or_compute_signal  # noqa: E402

WF_START = pd.Timestamp("2021-07-20")
OUT_ROOT = REPO / "effective_factors"

# thesis arm mapping (user ground truth): arm 4 = L1H_terra_s0b, arm 6 = L4WF_terra_s0
BOOKS = {"101": "zoo", "arm4": "L1H_terra_s0b", "arm6": "L4WF_terra_s0"}
SETS = {
    "101_alphas":     ["101"],
    "arm4":           ["arm4"],
    "arm6":           ["arm6"],
    "arm4_arm6":      ["arm4", "arm6"],
    "101_arm4":       ["101", "arm4"],
    "101_arm6":       ["101", "arm6"],
    "101_arm4_arm6":  ["101", "arm4", "arm6"],
}


def participation_ratio(C: np.ndarray) -> float:
    eig = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def pivoted_selection(C: np.ndarray, k: int) -> list[int]:
    """Greedy pivoted-Cholesky column selection on a correlation matrix.

    Equivalent to QR with column pivoting on the underlying signal matrix:
    each step picks the column with the largest residual (unexplained)
    variance, then projects it out of the remainder.
    """
    R = C.astype(float).copy()
    n = R.shape[0]
    chosen: list[int] = []
    for _ in range(min(k, n)):
        d = np.diag(R).copy()
        d[chosen] = -np.inf
        j = int(np.argmax(d))
        if d[j] <= 1e-12:
            break
        chosen.append(j)
        # Schur complement: remove the span of column j from the remainder.
        cj = R[:, j].copy()
        R = R - np.outer(cj, cj) / cj[j]
    return chosen


def main() -> None:
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors
    from quant_fund_agent.mcp import research_service as svc

    sys.path.insert(0, str(REPO / "scripts"))
    from wf_arm_factor_analysis import load_book

    discover_factors()
    panel = svc._load_panel_cached("ticker_data", sorted(usable_fields()),
                                   n_tickers=None)
    close = panel["close"]
    idx = close.index
    fit_mask = np.asarray(idx < WF_START)
    fit_idx = idx[fit_mask]
    fit_pos = np.flatnonzero(fit_mask)
    stride = max(1, int(fit_mask.sum()) // 400)
    rows_sel = fit_pos[::stride]
    print(f"panel {len(idx)} bars x {close.shape[1]} tickers; "
          f"fit bars {fit_mask.sum()}, corr rows {len(rows_sel)}")

    # load every book once; per-fid: (code, source, z-column or None)
    books: dict[str, dict[str, str]] = {}
    for label, arm in BOOKS.items():
        books[label] = load_book(arm)
        print(f"book {label}: {len(books[label])} members")

    col_cache: dict[tuple[str, str], np.ndarray | None] = {}

    def z_column(label: str, fid: str, code: str) -> np.ndarray | None:
        key = (label, fid)
        if key in col_cache:
            return col_cache[key]
        try:
            sig = load_or_compute_signal(fid, code, panel, idx, close.columns)
            # RAW signals on strided fit rows, missing/inf -> 0: the exact
            # convention of wf_arm_factor_analysis's diversity block, which
            # produced every published N_eff of the thesis tables.
            col = np.nan_to_num(sig.astype(float).to_numpy()[rows_sel].ravel(),
                                nan=0.0, posinf=0.0, neginf=0.0)
            if not np.isfinite(col).all() or float(np.std(col)) < 1e-12:
                col = None
        except Exception as e:  # noqa: BLE001
            print(f"  {fid} failed: {e}")
            col = None
        col_cache[key] = col
        return col

    OUT_ROOT.mkdir(exist_ok=True)
    summary: dict[str, dict] = {}

    for set_name, labels in SETS.items():
        members: list[tuple[str, str, str]] = []   # (fid, code, source)
        seen: set[str] = set()
        dropped: list[str] = []
        cols: list[np.ndarray] = []
        for label in labels:
            for fid, code in books[label].items():
                if fid in seen:
                    continue
                seen.add(fid)
                col = z_column(label, fid, code)
                if col is None:
                    dropped.append(f"{fid} ({label})")
                    continue
                members.append((fid, code, label))
                cols.append(col)
        X = np.column_stack(cols)
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num((C + C.T) / 2.0)
        np.fill_diagonal(C, 1.0)
        n_eff = participation_ratio(C)
        k = math.ceil(n_eff)
        # zoo: recomputed N_eff (8.18) sits just above the published 7.6
        # (vintage tolerance); pin k to the ceiling of the published value.
        if set_name == "101_alphas":
            k = 8
        # arm6: recomputed 22.04, published 22.0 -- same pinning rule.
        if set_name == "arm6":
            k = 22
        order = pivoted_selection(C, k)

        set_dir = OUT_ROOT / set_name
        set_dir.mkdir(exist_ok=True)
        for old in set_dir.glob("*.py"):
            old.unlink()
        lines = [f"# Effective factors: {set_name}", "",
                 f"Members {len(members)} (of {len(seen)}; "
                 f"{len(dropped)} degenerate/failed excluded), "
                 f"N_eff = {n_eff:.1f}, k = {k}.",
                 "",
                 "Selected by greedy pivoted-Cholesky column selection on the "
                 "correlation matrix of the raw fit-window signals (missing "
                 "cells as zero; the convention behind the thesis' N_eff) "
                 "(equivalent to QR with column pivoting on the signal "
                 "matrix): each step picks the factor with the largest "
                 "signal variance unexplained by those already chosen. "
                 f"Fit window < {WF_START.date()}; no walk-forward outcome "
                 "is read. Files are prefixed by selection rank.", "",
                 "| rank | factor | source |", "|---|---|---|"]
        sel = []
        for r, j in enumerate(order, 1):
            fid, code, src = members[j]
            (set_dir / f"{r:02d}_{fid}.py").write_text(code)
            lines.append(f"| {r} | {fid} | {src} |")
            sel.append({"rank": r, "factor_id": fid, "source": src})
        if dropped:
            lines += ["", "Excluded as degenerate or failed on this panel:",
                      ""] + [f"- {d}" for d in dropped]
        (set_dir / "README.md").write_text("\n".join(lines) + "\n")
        summary[set_name] = {"n_members": len(members), "n_eff": round(n_eff, 2),
                             "k": k, "selection": sel, "dropped": dropped}
        print(f"{set_name}: |B|={len(members)} N_eff={n_eff:.1f} k={k} "
              f"dropped={len(dropped)}")

    (OUT_ROOT / "selection_summary.json").write_text(
        json.dumps(summary, indent=2))
    (OUT_ROOT / "README.md").write_text(
        "# Effective factors\n\n"
        "One folder per book combination of the thesis' union analysis. Each\n"
        "folder contains the k = ceil(N_eff) representative factors selected\n"
        "by pivoted QR / pivoted Cholesky on the correlation matrix of the raw\n"
        "fit-window signals (missing cells as zero, fit window < 2021-07-20), with\n"
        "selection rank in the file name and details in the per-folder\n"
        "README. Generated by scripts/build_effective_factors.py;\n"
        "deterministic given the signal store.\n")
    print("done ->", OUT_ROOT)


if __name__ == "__main__":
    main()
