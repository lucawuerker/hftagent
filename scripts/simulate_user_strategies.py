"""Simulate product user strategies from the factor catalog.

For each persona (a plausible customer prompt), deterministically select a
GENEROUS factor set from the catalog (variable size, lean large), sanity-prune
with a LOOSE lasso + correlation check, then run the exact combined-book
protocol from scripts/backtest_combined_book.py: 4-model race on VAL with
trial counting + CSCV PBO, winner refit on the full in-panel window, ONE
forward pass on 2024-07 -> 2026-07. Trial accounting: 4 model trials per
persona + 1 for the lasso-prune choice (extra_trials=1); the catalog's global
n_trials is reported alongside for the factor-search layer.

Outputs data/books/catalog_<name>/sample_strategies/<persona>_report.json.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("simulate_user_strategies")

# Persona = what a customer might type + deterministic catalog filters.
PERSONAS = [
    {
        "key": "aggressive_short_term",
        "prompt": "I want an aggressive short-term trading strategy that turns "
                  "over quickly and squeezes alpha out of market microstructure.",
        "categories": ["microstructure", "statistical_arbitrage", "momentum"],
        "max_factors": 20,
    },
    {
        "key": "defensive_low_turnover",
        "prompt": "Something defensive for my retirement account — steady, "
                  "low-churn, based on company quality and value, not hype.",
        "categories": ["mean_reversion", "carry", "other", "volatility"],
        "prefer_fundamental": True,
        "max_factors": 14,
    },
    {
        "key": "earnings_events",
        "prompt": "I believe money is made around earnings — build me a "
                  "strategy that trades earnings surprises and analyst behaviour.",
        "keywords": ["earnings", "eps", "analyst", "surprise", "estimate",
                     "corporate", "valuation"],
        "max_factors": 16,
    },
    {
        "key": "diversified_all_weather",
        "prompt": "Give me your most diversified strategy — many independent "
                  "signals, nothing dominating, robust in every regime.",
        "categories": None,        # everything eligible
        "diversify": True,
        "max_factors": 24,
    },
    {
        "key": "contrarian_dip_buyer",
        "prompt": "I like buying panic and selling euphoria. Build me a "
                  "contrarian strategy that fades overreactions.",
        "categories": ["mean_reversion", "microstructure"],
        "keywords": ["reversal", "snapback", "reversion", "overreaction",
                     "dislocation", "fade", "exhaustion"],
        "max_factors": 18,
    },
]

FUNDAMENTAL_HINTS = ("eps", "revenue", "roe", "roa", "margin", "debt", "valuation",
                     "earnings", "marketcap", "cash", "capex", "dividend")


def select_factors(catalog: dict, persona: dict) -> list[dict]:
    """Deterministic, generous selection from catalog rows for one persona."""
    rows = catalog["factors"]
    cats = persona.get("categories")
    kws = [k.lower() for k in persona.get("keywords", [])]

    def matches(r: dict) -> bool:
        ok = True
        if cats is not None:
            ok = r["category"] in cats
        if kws:
            text = " ".join([r.get("trading_idea", ""), r.get("name", ""),
                             r.get("factor_id", ""), r.get("mechanism", "")]).lower()
            kw_hit = any(k in text for k in kws)
            ok = (ok and kw_hit) if cats is not None else kw_hit
        if persona.get("prefer_fundamental"):
            text = (r.get("trading_idea", "") + r.get("factor_id", "")).lower()
            ok = ok or any(h in text for h in FUNDAMENTAL_HINTS)
        return ok

    hits = [r for r in rows if matches(r)]
    hits.sort(key=lambda r: -(r.get("score") or 0.0))
    if persona.get("diversify"):
        # round-robin across categories so no bucket dominates
        import collections, itertools
        by_cat = collections.defaultdict(list)
        for r in hits:
            by_cat[r["category"]].append(r)
        hits = [r for r in itertools.chain.from_iterable(
            itertools.zip_longest(*by_cat.values())) if r is not None]
    return hits[: persona["max_factors"]]


def lasso_prune(signals: dict, close, is_mask, keep_min: int = 6) -> list[str]:
    """LOOSE redundancy check: drop a factor only when lasso zeroes it AND it
    correlates |rho|>=0.9 with a retained factor. Never below keep_min."""
    import numpy as np
    from sklearn.linear_model import LassoCV

    ids = list(signals)
    X_cols, y = [], None
    from quant_fund_agent.backtesting.data_loader import forward_returns
    fwd = forward_returns(close, horizon=6)
    zs = {}
    for fid in ids:
        s = signals[fid].reindex(index=close.index, columns=close.columns)
        mu, sd = s[is_mask].mean(), s[is_mask].std().replace(0, np.nan)
        zs[fid] = ((s - mu) / sd)
    rows = is_mask
    X = np.column_stack([np.nan_to_num(zs[f][rows].to_numpy().ravel()) for f in ids])
    yv = np.nan_to_num(fwd[rows].to_numpy().ravel())
    try:
        las = LassoCV(cv=3, n_alphas=20, max_iter=2000).fit(X, yv)
        zeroed = {ids[i] for i, c in enumerate(las.coef_) if c == 0.0}
    except Exception:  # noqa: BLE001 — lasso is advisory only
        return ids
    kept, dropped = [], []
    corr = np.corrcoef(X, rowvar=False)
    for i, fid in enumerate(ids):
        if fid not in zeroed:
            kept.append(fid)
    for i, fid in enumerate(ids):
        if fid in zeroed:
            redundant = any(abs(corr[i, ids.index(k)]) >= 0.9 for k in kept)
            (dropped if redundant else kept).append(fid)
    if len(kept) < keep_min:
        kept = ids
    return [f for f in ids if f in set(kept)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="quant.config.nasdaq100_2010_forward.yaml")
    p.add_argument("--catalog", default="nasdaq100_v1")
    p.add_argument("--personas", default=None, help="comma list of persona keys (default all)")
    p.add_argument("--construction", default="cross_sectional")
    p.add_argument("--extra-trials", type=int, default=1)
    args = p.parse_args()

    os.environ["QF_CONFIG_FILE"] = args.config
    os.environ.setdefault("QF_USE_MCP", "0")

    import numpy as np
    import pandas as pd

    from backtest_combined_book import PANEL_END, VAL_YEARS, run_book_backtest
    from quant_fund_agent.data import usable_fields
    from quant_fund_agent.factors import discover_factors, get_factor_class
    from quant_fund_agent.factors.inmem import compile_factor, compute_signal
    from quant_fund_agent.mcp import research_service as svc

    cat_dir = REPO / "data" / "books" / f"catalog_{args.catalog}"
    catalog = json.loads((cat_dir / "catalog.json").read_text())
    # code lives in the run states; rebuild the id->code map from the preruns
    codes: dict[str, str] = {}
    for prerun in catalog["preruns"]:
        st = json.loads((REPO / "data/workspaces" / catalog["config"] / "preruns"
                         / prerun.strip() / "evolution/state.json").read_text())
        for eg in st.get("kept_pool", []):
            prog = eg["genome"]["programs"][0]
            codes.setdefault(prog["factor_id"], prog["code"])

    fields = sorted(usable_fields())
    panel = svc._load_panel_cached("ticker_data", fields, n_tickers=None)
    close = panel["close"]
    idx = close.index
    forward_start = pd.Timestamp(PANEL_END)
    val_start = forward_start - pd.DateOffset(years=VAL_YEARS)
    is_mask = np.asarray(idx < val_start)
    discover_factors()

    only = set(args.personas.split(",")) if args.personas else None
    out_root = cat_dir / ("sample_strategies" if args.construction == "cross_sectional" else f"sample_strategies_{args.construction}")
    summary = []
    for persona in PERSONAS:
        if only and persona["key"] not in only:
            continue
        chosen = select_factors(catalog, persona)
        log.info("[%s] selected %d factors from catalog", persona["key"], len(chosen))
        sigs = {}
        for r in chosen:
            fid = r["factor_id"]
            try:
                cls = get_factor_class(fid) or compile_factor(codes[fid], fid)
                sigs[fid] = compute_signal(cls, panel)
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] %s failed (%s)", persona["key"], fid, e)
        pruned_ids = lasso_prune(sigs, close, is_mask)
        log.info("[%s] lasso-prune kept %d/%d", persona["key"], len(pruned_ids), len(sigs))
        report = run_book_backtest(
            pruned_ids, codes, out_root,
            (persona["key"] if args.construction == "cross_sectional"
             else f"{persona['key']}_{args.construction}"),
            extra_trials=args.extra_trials, construction=args.construction)
        report["persona_prompt"] = persona["prompt"]
        report["selected_before_prune"] = len(chosen)
        report["catalog_n_trials_global"] = catalog["n_trials_global"]
        (out_root / f"{report['label']}_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        summary.append({
            "persona": persona["key"], "n_factors": report["n_factors"],
            "winner": report["winner"],
            "val_ic": report["race"][report["winner"]]["val_ic"],
            "fwd_sharpe": report["forward"]["sharpe"],
            "fwd_ic": report["forward"]["ic"],
            "dsr": report["forward"]["deflated_sharpe_prob"],
            "pbo_val": report["pbo_cscv_val"],
        })
        print(json.dumps(summary[-1], indent=2, default=str))
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nALL PERSONAS DONE ->", out_root / "summary.json")


if __name__ == "__main__":
    main()
