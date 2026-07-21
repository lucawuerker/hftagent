"""The shared final *publish* filter — selection-time multiple-testing deflation.

Deflation used to be a per-candidate hard gate inside the fitness harness, which
(a) throttled the very throughput a "discovery" run wants and (b) double-counted
against the Pareto selection.  Following Bailey & López de Prado (deflate *once*,
at the end, on the full trial count), deflation now lives here — one choke point
run over the **final book**, whatever produced it (the Pareto archive, the
kept-pool, a greedy/elastic-net curation, or reserved mechanism-group archives).

Two things make this correct where a naive "``deflated_ic(|val_ic|)`` per factor"
would be wrong:

* **Deflate the statistic selection actually uses.**  We deflate the selected
  book's **combined-model OOS IC** (and report its Deflated Sharpe / PBO), not any
  single factor's standalone IC.  A complementary / conditioning factor has weak
  standalone IC but strong *marginal* value; standalone deflation would wrongly
  kill exactly the factors the reward is designed to find.
* **Prune by marginal (LOCO) contribution, never standalone IC.**  When the book
  fails deflation and must be narrowed, we drop the member whose *marginal*
  contribution to the combined model is smallest (a parasitic / negative-marginal
  factor), re-score, and repeat.  Dropping a parasite can *raise* the combined IC
  and push the book over the deflation bar; dropping a high-marginal factor never
  helps, so such a factor is retained.

Modes: ``off`` = discovery (keep everything, deflation reported but not enforced);
``on`` = validation (narrow the book to what beats selection luck for ``n_trials``).
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_fund_agent.research_eval import deflation
from quant_fund_agent.research_eval.splits import ThreeWaySplit

log = logging.getLogger("research_eval.publish")

PUBLISH_MODES = ("off", "on")


def _combined_val_ic(
    members: list[str], signals: Mapping[str, pd.DataFrame],
    close: pd.DataFrame, cfg: Any, split: ThreeWaySplit, model: str,
) -> tuple[float | None, int]:
    """Combined-model VAL IC + scored-obs count for ``members`` (IS-fit / VAL-score)."""
    from quant_fund_agent.research_eval.harness import _combined_prediction, _pooled_ic

    if not members:
        return None, 0
    pred = _combined_prediction([signals[m] for m in members], close,
                                split.is_mask, cfg, model)
    if pred is None:
        return None, 0
    return _pooled_ic(pred, close, cfg.target_horizon, split.val_mask,
                      split.is_mask, split.is_val_mask)


def publish_filter(
    signals: Mapping[str, pd.DataFrame],
    close: pd.DataFrame,
    cfg: Any,
    split: ThreeWaySplit,
    n_trials: int,
    *,
    mode: str = "on",
    marginal_model: str = "gradient_boosting",
    min_members: int = 1,
) -> dict[str, Any]:
    """Apply selection-time deflation to a final book → the published subset.

    Returns ``{"kept_factor_ids", "passed", "combined_ic", "combined_n_obs",
    "deflated", "n_trials", "mode", "dropped"}``.  ``passed`` is whether the kept
    book's combined deflated-IC t-stat is > 0 (always ``True`` in ``off`` mode).
    """
    ids = list(signals)
    if not ids:
        return {"kept_factor_ids": [], "passed": True, "combined_ic": None,
                "combined_n_obs": 0, "deflated": None, "n_trials": int(n_trials),
                "mode": mode, "dropped": []}

    ic, n = _combined_val_ic(ids, signals, close, cfg, split, marginal_model)
    defl = deflation.deflated_ic(abs(ic) if ic is not None else None,
                                 int(n), int(n_trials))

    def _passes(d: dict[str, Any]) -> bool:
        return d.get("deflated_t") is not None and d["deflated_t"] > 0

    # ── discovery mode: keep everything, deflation reported not enforced ──
    if mode != "on":
        return {"kept_factor_ids": ids, "passed": True, "combined_ic": ic,
                "combined_n_obs": int(n), "deflated": defl, "n_trials": int(n_trials),
                "mode": mode, "dropped": []}

    # ── validation mode: narrow the book to what beats selection luck ──
    kept = list(ids)
    dropped: list[str] = []
    while not _passes(defl) and len(kept) > max(1, int(min_members)):
        base_ic, _ = _combined_val_ic(kept, signals, close, cfg, split, marginal_model)
        base_ic = base_ic if base_ic is not None else 0.0
        # marginal (LOCO) contribution of each member = combined(kept) − combined(kept∖m)
        marg: dict[str, float] = {}
        for m in kept:
            sub = [x for x in kept if x != m]
            sub_ic, _ = _combined_val_ic(sub, signals, close, cfg, split, marginal_model)
            marg[m] = base_ic - (sub_ic if sub_ic is not None else 0.0)
        worst = min(marg, key=lambda m: marg[m])
        # every remaining member earns its place but the book still fails deflation →
        # no subset can be rescued (dropping a positive-marginal factor only lowers
        # the combined IC), so stop and report the honest failure.
        if marg[worst] > 0:
            break
        kept.remove(worst)
        dropped.append(worst)
        ic, n = _combined_val_ic(kept, signals, close, cfg, split, marginal_model)
        defl = deflation.deflated_ic(abs(ic) if ic is not None else None,
                                     int(n), int(n_trials))

    return {"kept_factor_ids": kept, "passed": _passes(defl), "combined_ic": ic,
            "combined_n_obs": int(n), "deflated": defl, "n_trials": int(n_trials),
            "mode": mode, "dropped": dropped}
