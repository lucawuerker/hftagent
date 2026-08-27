"""Shared helpers for the WF post-analysis scripts.

Signal store: every computed factor signal (full WF panel, float32) is
persisted once as parquet under
``data/comparisons/wf_arm_analysis/signal_store/<fid>__<codehash>.parquet``
and loaded from there by any later job — the expensive signal phase of the
big union runs and any future analysis never has to recompute a signal.
Keyed by (factor_id, sha1(code)) so identically-named factors from
different arms can never collide.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# QF_SIGNAL_STORE_DIR overrides the store location (the cache is keyed by
# (factor_id, sha1(code)) ONLY — not by panel — so any job on a different
# panel/universe MUST point this at its own store or it will silently reuse
# signals computed on another universe).  Unset -> original path, byte-identical.
SIGNAL_STORE = Path(os.environ.get("QF_SIGNAL_STORE_DIR")
                    or (REPO / "data/comparisons/wf_arm_analysis/signal_store"))


def signal_key(fid: str, code: str) -> str:
    h = hashlib.sha1(code.encode()).hexdigest()[:10]
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in fid)[:80]
    return f"{safe}__{h}"


def load_or_compute_signal(fid, code, panel, idx, cols):
    """float32 signal frame aligned to (idx, cols); parquet-cached."""
    import numpy as np
    import pandas as pd

    from quant_fund_agent.factors import get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal

    SIGNAL_STORE.mkdir(parents=True, exist_ok=True)
    p = SIGNAL_STORE / f"{signal_key(fid, code)}.parquet"
    if p.exists():
        sig = pd.read_parquet(p).reindex(index=idx, columns=cols)
        return sig.astype("float32")
    cls = get_factor_class(fid) or compile_factor(code, fid)
    sig = compute_signal(cls, panel).reindex(
        index=idx, columns=cols).astype("float32")
    if float(np.isfinite(sig.to_numpy()).mean()) < 0.01:
        raise ValueError("degenerate coverage")
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    sig.to_parquet(tmp)
    os.replace(tmp, p)  # atomic: concurrent writers just win-once
    return sig
