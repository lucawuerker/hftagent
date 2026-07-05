"""End-of-run book curation — the "curate once, at the end" half of Lever 2.

The two-stage alternative to letting Pareto domination decide the final book:
during the search *every gate-passing factor is kept* (the controller's
``kept_pool``); at persist time one of these deterministic curators turns that
pool into the deployed book.  This decouples *what survives selection* (the
Pareto archive, still the parent-selection + marginal-scoring reference) from
*what we keep* (chosen here), so a factor is no longer discarded merely for being
dominated — the pathology that made the discovered set collapse.

Two modes, both signal-based (take signal frames, no LLM, no files) and scored on
the **same** IS-fit / VAL-score seam as the fitness harness so research and
curation cannot drift:

* ``greedy`` — forward selection maximising the combined model's VAL IC.  Add the
  factor whose *marginal* combined-IC gain is largest; stop at ``n_keep`` or when
  no remaining factor improves the combined signal (``min_gain``).  This is the
  direct "does it add value in combination" test and auto-sizes the book.
* ``elastic_net`` — stability selection: fit an elastic net on the standardised
  factor signals → forward returns over many random sub-samples of the
  development rows, and keep the factors selected (non-zero coefficient) in at
  least ``threshold`` of them (or the top ``n_keep`` by selection frequency).
  Overfitting-aware and it measures "survives when combined" directly.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_fund_agent.research_eval.splits import ThreeWaySplit

log = logging.getLogger("research_eval.curation")

CURATION_MODES = ("archive", "greedy", "elastic_net")


def greedy_book(
    signals: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    cfg: Any,
    split: ThreeWaySplit,
    *,
    n_keep: int | None = None,
    min_gain: float = 0.0,
    model: str = "ridge",
) -> dict[str, Any]:
    """Forward-select factors that raise the combined model's VAL IC.

    Returns ``{"kept_factor_ids", "combined_ic", "trace"}``.  The first factor
    (the best single) is always kept; each subsequent factor must lift the
    combined IC by more than ``min_gain`` (ignored when ``n_keep`` is set, in
    which case exactly the best ``n_keep`` are taken).
    """
    from quant_fund_agent.research_eval.harness import _combined_prediction, _pooled_ic

    ids = list(signals)
    h = cfg.target_horizon
    is_mask, val_mask, avail = split.is_mask, split.val_mask, split.is_val_mask
    limit = len(ids) if n_keep is None else max(1, min(int(n_keep), len(ids)))

    selected: list[str] = []
    remaining = list(ids)
    current_ic = 0.0
    trace: list[dict[str, Any]] = []

    while remaining and len(selected) < limit:
        best_fid: str | None = None
        best_ic: float | None = None
        for fid in remaining:
            feats = [signals[s] for s in selected] + [signals[fid]]
            pred = _combined_prediction(feats, close, is_mask, cfg, model)
            if pred is None:
                continue
            ic = _pooled_ic(pred, close, h, val_mask, is_mask, avail)[0]
            if ic is None:
                continue
            if best_ic is None or ic > best_ic:
                best_ic, best_fid = ic, fid
        if best_fid is None:
            break
        gain = best_ic - current_ic
        # stop only once we already hold at least one factor and the best remaining
        # addition no longer improves the combined signal (free book-sizing).
        if n_keep is None and selected and gain <= min_gain:
            trace.append({"factor_id": best_fid, "combined_ic": best_ic,
                          "gain": gain, "kept": False})
            break
        selected.append(best_fid)
        remaining.remove(best_fid)
        current_ic = best_ic
        trace.append({"factor_id": best_fid, "combined_ic": best_ic,
                      "gain": gain, "kept": True})

    return {"kept_factor_ids": selected, "combined_ic": current_ic, "trace": trace}


def elastic_net_book(
    signals: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    cfg: Any,
    split: ThreeWaySplit,
    *,
    n_keep: int | None = None,
    threshold: float = 0.6,
    n_resamples: int = 30,
    l1_ratio: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Stability selection with an elastic net → the reliably-selected factors.

    Fit an :class:`~sklearn.linear_model.ElasticNetCV` on the standardised factor
    signals → forward returns over ``n_resamples`` random halves of the
    development rows; ``selection_frequency`` is the fraction of fits keeping each
    factor (non-zero coefficient).  Keep factors with frequency ≥ ``threshold``
    (or the top ``n_keep`` by frequency).  Falls back to keeping all factors when
    there is too little data / too few members to fit.
    """
    from quant_fund_agent.backtesting.data_loader import forward_returns
    from quant_fund_agent.research_eval.harness import _flat_z, _label_available_mask

    ids = list(signals)
    if len(ids) < 2:
        return {"kept_factor_ids": ids, "selection_frequency": {i: 1.0 for i in ids}}

    h = cfg.target_horizon
    n_cols = close.shape[1]
    X = np.column_stack([
        _flat_z(signals[fid], close, stat_mask=split.is_mask) for fid in ids])
    y = forward_returns(close, h).to_numpy(dtype=float).ravel()
    dev_rows = np.repeat(np.asarray(split.is_val_mask) & _label_available_mask(
        split.is_val_mask, h), n_cols)
    finite = dev_rows & np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xf = X[finite]
    yf = y[finite]
    if len(yf) < 50:
        log.info("[curate:elastic_net] too few dev rows (%d) — keeping all", len(yf))
        return {"kept_factor_ids": ids, "selection_frequency": {i: 1.0 for i in ids}}

    try:
        from sklearn.linear_model import ElasticNetCV
    except Exception as e:  # noqa: BLE001 — sklearn always present, but degrade safely
        log.warning("[curate:elastic_net] sklearn unavailable (%s) — keeping all", e)
        return {"kept_factor_ids": ids, "selection_frequency": {i: 1.0 for i in ids}}

    rng = np.random.default_rng(seed)
    counts = np.zeros(len(ids), dtype=float)
    n_fits = 0
    half = max(30, len(yf) // 2)
    for _ in range(max(1, n_resamples)):
        sub = rng.choice(len(yf), size=half, replace=False)
        try:
            try:
                en = ElasticNetCV(l1_ratio=l1_ratio, alphas=10, cv=3,
                                  max_iter=2000, random_state=0)
            except TypeError:  # older sklearn: `alphas` expects an array, use n_alphas
                en = ElasticNetCV(l1_ratio=l1_ratio, n_alphas=10, cv=3,
                                  max_iter=2000, random_state=0)
            en.fit(Xf[sub], yf[sub])
        except Exception as e:  # noqa: BLE001 — a degenerate resample must not kill the run
            log.debug("[curate:elastic_net] resample fit failed: %s", e)
            continue
        counts += np.abs(en.coef_) > 1e-8
        n_fits += 1

    if n_fits == 0:
        return {"kept_factor_ids": ids, "selection_frequency": {i: 1.0 for i in ids}}
    freq = counts / n_fits
    order = list(np.argsort(-freq))
    if n_keep is not None:
        kept_idx = order[: max(1, int(n_keep))]
    else:
        kept_idx = [i for i in order if freq[i] >= threshold] or [order[0]]
    return {
        "kept_factor_ids": [ids[i] for i in kept_idx],
        "selection_frequency": {ids[i]: round(float(freq[i]), 3) for i in range(len(ids))},
    }


def curate(
    mode: str,
    signals: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    cfg: Any,
    split: ThreeWaySplit,
    *,
    n_keep: int | None = None,
    min_gain: float = 0.0,
    model: str = "ridge",
    threshold: float = 0.6,
    n_resamples: int = 30,
    l1_ratio: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """Dispatch to the requested curation mode → ``{"kept_factor_ids", ...}``."""
    if not signals:
        return {"kept_factor_ids": [], "mode": mode}
    if mode == "greedy":
        out = greedy_book(signals, close, cfg, split, n_keep=n_keep,
                          min_gain=min_gain, model=model)
    elif mode == "elastic_net":
        out = elastic_net_book(signals, close, cfg, split, n_keep=n_keep,
                              threshold=threshold, n_resamples=n_resamples,
                              l1_ratio=l1_ratio, seed=seed)
    else:
        raise ValueError(f"unknown curation mode {mode!r} (use one of {CURATION_MODES})")
    out["mode"] = mode
    return out
