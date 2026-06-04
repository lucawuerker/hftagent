"""Showcase — Stage 4: Portfolio Manager Agent (no market data required).

Unlike Stages 1–3 (which run the real agent and *stop* right before the
market-data step), the Portfolio Manager has no creative step that precedes the
data: its whole job is to allocate capital across strategies that have **already
been backtested and approved**.  Crucially, the PM never opens the proprietary
``ticker_data/`` panel at all — it reasons purely over the **strategy book**:

  * each strategy's summary **backtest metrics** (Sharpe, drawdown, vol, …), and
  * each strategy's persisted **return series** (used to estimate the
    cross-strategy correlation / covariance matrix).

Both of those already live in the repo — the strategy database
(``data/strategies/strategy_db.json``) and the per-strategy return CSVs under
``data/strategies/returns/`` — so this stage runs the **real PM agent,
end-to-end and unmodified, over the real strategy book** with no market data.

    load_universe → monitor_performance → screen_strategies
                  → construct_portfolio → finalise

In its default **SELECTOR** mode the PM is fully deterministic and needs **no
LLM and no network at all** (monitoring uses the personality's rule-based
thresholds; screening and construction are pure quant maths).  ``--mode active``
(or ``--committee --voting llm_moderator``) instead lets an LLM drive the
monitor / screen / construction-method choices, which needs an ``OPENAI_API_KEY``.

The repo's strategy book is a set of strong, healthy strategies, so the
*monitor* node keeps them all.  To also see the monitor **flag / retire**
degraded strategies, pass ``--synthetic`` to run over a hand-built book that
includes a drifting and a broken strategy (returns generated, no market data).

Usage::

    ./venv/bin/python showcase_pipeline/4_portfolio_manager.py
    ./venv/bin/python showcase_pipeline/4_portfolio_manager.py --personality defensive
    ./venv/bin/python showcase_pipeline/4_portfolio_manager.py --committee
    ./venv/bin/python showcase_pipeline/4_portfolio_manager.py --synthetic   # degraded book → see flag/retire
    ./venv/bin/python showcase_pipeline/4_portfolio_manager.py --mode active  # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from dataclasses import dataclass, field

import showcase_common as sc  # FIRST import: configures env + working dir

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)-22s  %(message)s",
    datefmt="%H:%M:%S",
)

# The synthetic book (``--synthetic``) generates *daily* return bars purely so
# the numbers read nicely (annual Sharpe ≈ return / vol).  The real book trades
# 10-second bars, so its annualisation factor is pipeline.BARS_PER_YEAR.
TRADING_DAYS = 252
ANNUALISATION_FACTOR_DAILY = float(TRADING_DAYS)

# Common "risk factors" the synthetic strategies load on.  Shared loadings are
# what create realistic cross-strategy correlation (e.g. two momentum sleeves
# move together), which is exactly what the PM's diversification screen and
# risk-based construction need to have something to chew on.
COMMON_FACTORS = ("F_momentum", "F_meanrev", "F_vol", "F_macro")


# ---------------------------------------------------------------------------
# Real strategy book (default) — the approved strategies already in the repo
# ---------------------------------------------------------------------------

def load_real_book():
    """Load the real ``strategy_db.json`` + its persisted return series.

    Returns ``(db, annualisation_factor, source_label)``.  The DB is loaded
    fresh (no mutation of the on-disk file) and its records already point at the
    real return CSVs under ``data/strategies/returns/`` via ``returns_path``, so
    the correlation / covariance matrices are computed from real data — without
    ever touching the order-book / price panel.
    """
    from quant_fund_agent.databases import StrategyDatabase
    from quant_fund_agent.pipeline import BARS_PER_YEAR

    db = StrategyDatabase()
    db.load_from_json(sc.STRATEGY_DB_REAL)
    return db, float(BARS_PER_YEAR), f"real strategy book ({sc.STRATEGY_DB_REAL})"


# ---------------------------------------------------------------------------
# Synthetic strategy book (``--synthetic``) — used to demo flag / retire
# ---------------------------------------------------------------------------

@dataclass
class StratSpec:
    """One synthetic approved strategy.

    ``ann_return`` / ``ann_vol`` are the hand-set in-sample summary metrics the
    PM screens on (Sharpe = ann_return / ann_vol).  ``loadings`` map common
    factors → loading and control the correlation structure of the *generated*
    return series (two strategies sharing a loading become correlated).  When
    ``live`` is given the strategy is currently deployed (pm_status = LIVE) and
    those numbers feed the PM's live-performance monitor.
    """
    sid: str
    name: str
    category: str
    model_type: str
    factor_ids: list[str]
    ann_return: float
    ann_vol: float
    max_drawdown: float
    hit_rate: float
    loadings: dict[str, float] = field(default_factory=dict)
    live: dict[str, float] | None = None   # {sharpe, ann_return, ann_vol, max_dd, hit}
    note: str = ""

    @property
    def sharpe(self) -> float:
        return self.ann_return / self.ann_vol if self.ann_vol else 0.0


# The synthetic book is hand-designed so the deterministic SELECTOR-mode run
# tells a clear story: a broken live strategy gets retired, a drifting one gets
# flagged, two momentum sleeves are too correlated to both survive the
# diversification cap, and two strategies are simply too weak / too deep in
# drawdown to deploy.
_L_MOM = math.sqrt(0.90)    # → pairwise ρ ≈ 0.90 between the two momentum sleeves
_L_REV = math.sqrt(0.60)    # → pairwise ρ ≈ 0.60 between the two mean-reversion sleeves

SPECS: list[StratSpec] = [
    StratSpec(
        sid="mom_breakout", name="Momentum Breakout", category="momentum",
        model_type="random_forest", factor_ids=["alpha_012", "alpha_101", "vol_breakout"],
        ann_return=0.168, ann_vol=0.12, max_drawdown=-0.06, hit_rate=0.56,
        loadings={"F_momentum": _L_MOM},
        note="Strong, shallow-DD — the anchor of the book.",
    ),
    StratSpec(
        sid="mom_trend_follow", name="Trend Following", category="momentum",
        model_type="xgboost", factor_ids=["alpha_012", "ma_cross", "adx_trend"],
        ann_return=0.161, ann_vol=0.14, max_drawdown=-0.08, hit_rate=0.55,
        loadings={"F_momentum": _L_MOM},
        note="Good, but ρ≈0.90 with Momentum Breakout → diversification cap drops one.",
    ),
    StratSpec(
        sid="meanrev_intraday", name="Intraday Mean-Reversion", category="mean_reversion",
        model_type="ridge", factor_ids=["alpha_002", "rsi_revert"],
        ann_return=0.100, ann_vol=0.10, max_drawdown=-0.05, hit_rate=0.58,
        loadings={"F_meanrev": _L_REV},
        live={"sharpe": 0.85, "ann_return": 0.085, "ann_vol": 0.10, "max_dd": -0.06, "hit": 0.57},
        note="Currently LIVE and tracking its backtest → monitor keeps it.",
    ),
    StratSpec(
        sid="stat_arb_pairs", name="Stat-Arb Pairs", category="mean_reversion",
        model_type="ridge", factor_ids=["coint_spread", "ou_zscore"],
        ann_return=0.099, ann_vol=0.09, max_drawdown=-0.04, hit_rate=0.60,
        loadings={"F_meanrev": _L_REV},
        note="ρ≈0.60 with Intraday Mean-Reversion → under the cap, both can be held.",
    ),
    StratSpec(
        sid="vol_carry", name="Volatility Carry", category="volatility",
        model_type="static_weights", factor_ids=["vrp_carry", "term_structure"],
        ann_return=0.0638, ann_vol=0.11, max_drawdown=-0.07, hit_rate=0.52,
        loadings={"F_vol": 0.50},
        live={"sharpe": 0.05, "ann_return": 0.006, "ann_vol": 0.11, "max_dd": -0.13, "hit": 0.49},
        note="Marginal IS Sharpe AND drifting live → flagged for the research team.",
    ),
    StratSpec(
        sid="liquidity_provision", name="Liquidity Provision", category="microstructure",
        model_type="static_weights", factor_ids=["queue_imbalance", "spread_capture"],
        ann_return=0.056, ann_vol=0.08, max_drawdown=-0.04, hit_rate=0.62,
        loadings={"F_vol": 0.25},
        note="Low-vol diversifier — favoured by min-variance / defensive PMs.",
    ),
    StratSpec(
        sid="overnight_gap", name="Overnight Gap", category="event",
        model_type="random_forest", factor_ids=["gap_signal", "earnings_drift"],
        ann_return=0.0845, ann_vol=0.13, max_drawdown=-0.09, hit_rate=0.53,
        loadings={"F_macro": 0.40},
        note="Passes the balanced screen, just misses the stricter defensive one.",
    ),
    StratSpec(
        sid="news_sentiment", name="News Sentiment", category="alternative",
        model_type="xgboost", factor_ids=["nlp_sentiment", "headline_shock"],
        ann_return=0.040, ann_vol=0.20, max_drawdown=-0.12, hit_rate=0.50,
        loadings={},  # idiosyncratic — uncorrelated with the rest
        note="Sharpe 0.20 — below every personality's floor → screened out.",
    ),
    StratSpec(
        sid="macro_overlay", name="Macro Overlay", category="macro",
        model_type="ridge", factor_ids=["rates_carry", "fx_momentum"],
        ann_return=0.112, ann_vol=0.16, max_drawdown=-0.26, hit_rate=0.52,
        loadings={"F_macro": 0.50},
        live={"sharpe": -0.70, "ann_return": -0.112, "ann_vol": 0.16, "max_dd": -0.32, "hit": 0.44},
        note="LIVE but badly broken (live Sharpe -0.7) → monitor retires it.",
    ),
]


def _generate_returns(spec: StratSpec, factors: dict[str, np.ndarray],
                      index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """Generate a daily return series matching ``spec``'s vol + correlation.

    ``z = Σ_k loading_k · F_k + sqrt(1 − Σ loading_k²) · ε`` has unit variance,
    and two strategies' z-series correlate at ``Σ_k loading_ik · loading_jk``.
    We then scale to the target daily vol and shift to the target daily mean, so
    the realised annualised vol/return reproduce the hand-set summary metrics.
    """
    n = len(index)
    s = min(sum(v * v for v in spec.loadings.values()), 0.98)
    z = np.zeros(n)
    for fname, load in spec.loadings.items():
        z += load * factors[fname]
    z += math.sqrt(max(1.0 - s, 1e-6)) * rng.standard_normal(n)
    # Standardise empirically so the realised vol lands exactly on target
    # (preserves the correlation structure — it's just an affine rescale).
    z = (z - z.mean()) / z.std(ddof=1)

    sigma_d = spec.ann_vol / math.sqrt(TRADING_DAYS)
    mu_d = spec.ann_return / TRADING_DAYS
    daily = mu_d + sigma_d * z
    return pd.Series(daily, index=index, name=spec.sid)


def build_synthetic_book(seed: int, n_days: int):
    """Create a fresh showcase StrategyDatabase populated with the synthetic book.

    Returns ``(db, annualisation_factor, source_label)`` to mirror
    ``load_real_book``.
    """
    from quant_fund_agent.databases import StrategyDatabase
    from quant_fund_agent.schemas import (
        PMStatus,
        StrategyBacktestMetrics,
        StrategyRecord,
        StrategyStatus,
    )

    # Start from a clean slate every run: the book is deterministic, so there is
    # no state worth preserving between runs.
    if sc.PM_STRATEGY_DB_SHOWCASE.exists():
        sc.PM_STRATEGY_DB_SHOWCASE.unlink()
    if sc.PM_RETURNS_DIR_SHOWCASE.exists():
        shutil.rmtree(sc.PM_RETURNS_DIR_SHOWCASE)

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    factors = {name: rng.standard_normal(n_days) for name in COMMON_FACTORS}

    db = StrategyDatabase(returns_dir=sc.PM_RETURNS_DIR_SHOWCASE)
    for spec in SPECS:
        is_metrics = StrategyBacktestMetrics(
            annualised_return=spec.ann_return,
            annualised_volatility=spec.ann_vol,
            sharpe_ratio=round(spec.sharpe, 4),
            max_drawdown=spec.max_drawdown,
            hit_rate=spec.hit_rate,
        )
        live_metrics = None
        pm_status = PMStatus.NOT_DEPLOYED
        if spec.live is not None:
            pm_status = PMStatus.LIVE
            live_metrics = StrategyBacktestMetrics(
                annualised_return=spec.live["ann_return"],
                annualised_volatility=spec.live["ann_vol"],
                sharpe_ratio=spec.live["sharpe"],
                max_drawdown=spec.live["max_dd"],
                hit_rate=spec.live["hit"],
            )
        record = StrategyRecord(
            id=spec.sid,
            name=spec.name,
            class_name="ModelStrategy" if spec.model_type != "static_weights" else "DynamicStrategy",
            description=spec.note,
            factor_ids=spec.factor_ids,
            status=StrategyStatus.APPROVED,   # the statistician has signed off
            backtest_metrics=is_metrics,
            live_metrics=live_metrics,
            model_type=spec.model_type,
            pm_status=pm_status,
            metadata={
                "showcase": True,
                "category": spec.category,
                "note": "Synthetic approved strategy — metrics hand-set, returns "
                        "generated (no market data in this repository).",
            },
        )
        returns = _generate_returns(spec, factors, index, rng)
        db.register_strategy(record, portfolio_returns=returns)

    return db, ANNUALISATION_FACTOR_DAILY, "synthetic strategy book (generated)"


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_universe(db, source_label: str) -> None:
    recs = db.list_strategies()
    show_oos = any(r.oos_backtest_metrics for r in recs)
    show_live = any(r.live_metrics for r in recs)
    id_w = min(max((len(r.id) for r in recs), default=8), 52)

    sc.section(f"STRATEGY BOOK ({len(recs)} approved strategies)")
    print(f"  Source: {source_label} — NO market data is opened.")
    header = (f"  {'id':<{id_w}} {'IS Shrp':>8} {'IS Vol':>7} {'IS MaxDD':>9}")
    if show_oos:
        header += f" {'OOS Shrp':>9}"
    header += f" {'pm_status':>13}"
    if show_live:
        header += f" {'live Shrp':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rec in recs:
        m = rec.backtest_metrics
        row = (f"  {rec.id[:id_w]:<{id_w}} {m.sharpe_ratio:>8.2f} "
               f"{m.annualised_volatility:>7.1%} {m.max_drawdown:>9.2%}")
        if show_oos:
            om = rec.oos_backtest_metrics
            row += f" {om.sharpe_ratio:>9.2f}" if om else f" {'—':>9}"
        row += f" {rec.pm_status.value:>13}"
        if show_live:
            lv = rec.live_metrics
            row += f" {lv.sharpe_ratio:>10.2f}" if lv else f" {'—':>10}"
        print(row)


def _print_correlations(state) -> None:
    corr = state.correlation_matrix
    if corr is None or corr.empty:
        return
    sc.section("STRATEGY CORRELATION MATRIX (from the persisted return series)")
    # Long real IDs make a full matrix unreadable; abbreviate to id prefix+suffix.
    def short(sid: str) -> str:
        return sid if len(sid) <= 16 else f"{sid[:9]}…{sid[-4:]}"
    disp = corr.rename(index=short, columns=short)
    print(disp.round(2).to_string())


def _print_monitor(state) -> None:
    sc.section(f"MONITOR — live strategies reviewed ({len(state.monitor_actions)})")
    if not state.monitor_actions:
        print("  (no LIVE strategies to monitor)")
        return
    for a in state.monitor_actions:
        print(f"  • {a['strategy_id'][:40]:<40} → {a['action'].upper():<7} "
              f"{a.get('reasoning', '')}")
    if state.flagged_strategy_ids:
        print(f"\n  FLAGGED for research review : {state.flagged_strategy_ids}")
    if state.retired_strategy_ids:
        print(f"  RETIRED (removed from book) : {state.retired_strategy_ids}")


def _print_allocations(record) -> None:
    sc.section("FINAL ALLOCATION")
    print(f"  PM            : {record.pm_name}  "
          f"(mode={record.mode.value}, personality={record.personality.value})")
    print(f"  construction  : {record.construction_method.value}")
    enabled = [a for a in record.allocations if a.enabled]
    print(f"  {len(enabled)} strategies funded:")
    for a in enabled:
        print(f"      {a.strategy_id[:48]:<48} weight = {a.weight:>7.2%}")
    if record.expected_metrics:
        em = record.expected_metrics
        print("\n  Ex-ante portfolio metrics (annualised):")
        if "expected_return" in em:
            print(f"      expected return     : {em['expected_return']:>8.2%}")
        print(f"      expected volatility : {em.get('expected_volatility', float('nan')):>8.2%}")
        if "expected_sharpe" in em:
            print(f"      expected Sharpe     : {em['expected_sharpe']:>8.2f}")
        print(f"      gross / net exposure: {em.get('gross_exposure', float('nan')):.2f}"
              f" / {em.get('net_exposure', float('nan')):.2f}")


# ---------------------------------------------------------------------------
# Single-PM run (steps through the real graph nodes one at a time)
# ---------------------------------------------------------------------------

def run_single_pm(db, pdb, *, mode, personality, target_n, annualisation_factor,
                  source_label) -> object:
    from quant_fund_agent.agents.portfolio_manager.graph import (
        construct_portfolio_node,
        finalise,
        load_universe,
        monitor_performance,
        screen_strategies,
    )
    from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState
    from quant_fund_agent.portfolio import get_personality_profile

    profile = get_personality_profile(personality)
    state = PortfolioManagerState(
        pm_name=f"{personality.value}_pm",
        mode=mode,
        personality=personality,
        profile=profile,
        target_n_strategies=target_n,
        annualisation_factor=annualisation_factor,
        strategy_db=db,
        portfolio_db=pdb,
    )

    sc.section("PM CONFIGURATION")
    print(f"  personality : {profile.name}  —  {profile.description}")
    print(f"  mode        : {mode.value}  "
          f"({'LLM-driven monitor/screen/construct' if mode.value == 'active' else 'deterministic rule-based, no LLM'})")
    print(f"  screen      : min_sharpe={profile.min_sharpe}, "
          f"max_drawdown={profile.max_drawdown}, "
          f"min_hit_rate={profile.min_hit_rate}, "
          f"max_pairwise_corr={profile.max_pairwise_correlation}")
    print(f"  construction: default method = {profile.default_construction_method.value}, "
          f"target_n={target_n or profile.target_n_strategies}")

    # ── Step through the five production nodes, narrating each ──
    state = state.model_copy(update=load_universe(state))
    _print_universe(db, source_label)
    _print_correlations(state)

    state = state.model_copy(update=monitor_performance(state))
    _print_monitor(state)

    state = state.model_copy(update=screen_strategies(state))
    sc.section(f"SCREEN — roster picked ({len(state.selected_strategy_ids)})")
    for sid in state.selected_strategy_ids:
        rec = db.get_strategy(sid)
        print(f"  ✓ {sid[:48]:<48} ({rec.name})")
    print(f"\n  rationale: {state.screen_rationale}")

    state = state.model_copy(update=construct_portfolio_node(state))
    sc.section(f"CONSTRUCT — method = {state.chosen_method.value}")
    for sid, w in sorted(state.raw_weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {sid[:48]:<48} {w:>7.2%}")

    state = state.model_copy(update=finalise(state))
    _print_allocations(state.portfolio_record)
    return state.portfolio_record


# ---------------------------------------------------------------------------
# Committee run
# ---------------------------------------------------------------------------

def run_committee(db, pdb, *, mode, voting, target_n, annualisation_factor,
                  source_label) -> object:
    from quant_fund_agent.agents.portfolio_manager.committee import (
        CommitteeConfig,
        run_pm_committee,
    )
    from quant_fund_agent.agents.portfolio_manager.state import PortfolioManagerState
    from quant_fund_agent.portfolio import get_personality_profile
    from quant_fund_agent.schemas import PMPersonality

    personalities = [PMPersonality.DEFENSIVE, PMPersonality.BALANCED, PMPersonality.AGGRESSIVE]
    states = [
        PortfolioManagerState(
            pm_name=f"committee_{p.value}",
            mode=mode,
            personality=p,
            profile=get_personality_profile(p),
            target_n_strategies=target_n,
            annualisation_factor=annualisation_factor,
            strategy_db=db,
            portfolio_db=pdb,
        )
        for p in personalities
    ]

    sc.section("COMMITTEE CONFIGURATION")
    print(f"  {len(states)} PMs: {[p.value for p in personalities]}")
    print(f"  voting method : {voting.value}")
    print(f"  mode          : {mode.value}")

    # Show the book once up front (every PM votes on the same book).
    _print_universe(db, source_label)

    cfg = CommitteeConfig(voting_method=voting, pm_name="investment_committee")
    record = run_pm_committee(states, cfg, db, pdb)

    sc.section("PER-PM PROPOSALS")
    for prop in record.metadata.get("pm_proposals", []):
        print(f"\n  {prop['pm_name']} ({prop['personality']}, "
              f"method={prop['construction_method']}):")
        if prop["weights"]:
            for sid, w in sorted(prop["weights"].items(), key=lambda kv: -abs(kv[1])):
                print(f"      {sid[:48]:<48} {w:>7.2%}")
        else:
            print("      (no strategies passed this PM's screen)")
        if prop["flagged"]:
            print(f"      flagged: {prop['flagged']}")
        if prop["retired"]:
            print(f"      retired: {prop['retired']}")

    sc.section("CONSENSUS")
    if record.flagged_strategy_ids:
        print(f"  flagged by committee : {record.flagged_strategy_ids}")
    if record.retired_strategy_ids:
        print(f"  retired by committee : {record.retired_strategy_ids}")
    _print_allocations(record)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", action="store_true",
                   help="Run over a hand-built degraded book (returns generated, "
                        "no market data) instead of the real strategy_db.json — "
                        "lets you see the monitor flag/retire strategies.")
    p.add_argument("--mode", choices=("selector", "active"), default="selector",
                   help="selector = deterministic, no LLM (default); "
                        "active = LLM drives monitor/screen/construction.")
    p.add_argument("--personality", choices=("defensive", "balanced", "aggressive"),
                   default="balanced", help="Single-PM personality (ignored with --committee).")
    p.add_argument("--committee", action="store_true",
                   help="Run a 3-PM committee (defensive + balanced + aggressive) "
                        "and aggregate their proposals into one consensus.")
    p.add_argument("--voting", choices=("simple_average", "weighted_average", "llm_moderator"),
                   default="simple_average", help="Committee aggregation method.")
    p.add_argument("--target-n", type=int, default=None,
                   help="Override how many strategies the PM(s) want to deploy.")
    p.add_argument("--seed", type=int, default=42,
                   help="[--synthetic only] Seed for the generated return series.")
    p.add_argument("--days", type=int, default=2 * TRADING_DAYS,
                   help="[--synthetic only] Length of each generated series in trading days.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from quant_fund_agent.databases import PortfolioDatabase
    from quant_fund_agent.schemas import PMMode, PMPersonality, VotingMethod

    mode = PMMode(args.mode)
    voting = VotingMethod(args.voting)

    sc.banner("SHOWCASE · STAGE 4 · PORTFOLIO MANAGER")
    print("Runs the REAL PM agent over the strategy book — no market data, and")
    print("(in the default SELECTOR mode) no LLM either.  The PM monitors live")
    print("strategies, screens the universe, builds a diversified portfolio and")
    print("persists the allocation decision.")

    # Only the LLM-driven paths need an API key.
    if mode == PMMode.ACTIVE or (args.committee and voting == VotingMethod.LLM_MODERATOR):
        sc.require_openai_key()

    if args.synthetic:
        db, annualisation_factor, source_label = build_synthetic_book(
            seed=args.seed, n_days=args.days,
        )
    else:
        db, annualisation_factor, source_label = load_real_book()
        if not db.list_strategies():
            print(f"\nERROR: no strategies found in {sc.STRATEGY_DB_REAL}.\n"
                  "Run the research → architect → statistician pipeline first, or "
                  "use --synthetic for a self-contained demo.")
            return

    pdb = PortfolioDatabase()

    if args.committee:
        record = run_committee(db, pdb, mode=mode, voting=voting,
                               target_n=args.target_n,
                               annualisation_factor=annualisation_factor,
                               source_label=source_label)
    else:
        record = run_single_pm(db, pdb, mode=mode,
                               personality=PMPersonality(args.personality),
                               target_n=args.target_n,
                               annualisation_factor=annualisation_factor,
                               source_label=source_label)

    # ── Persist the showcase DBs so the outcome can be inspected on disk ──
    # We write to *copies*; the real strategy_db.json is only ever read.
    db.save_to_json(sc.PM_STRATEGY_DB_SHOWCASE)
    pdb.save_to_json(sc.PM_PORTFOLIO_DB_SHOWCASE)

    sc.section("PERSISTED PORTFOLIO RECORD (JSON)")
    print(json.dumps(record.model_dump(mode="json"), indent=2, default=str))

    print("\n" + "·" * 80)
    print("No market data was used: the PM reasons entirely over the strategy book")
    print("(metrics + persisted return series), never the order-book / price panel.")
    print("·" * 80)

    sc.banner("STAGE 4 COMPLETE")
    print(f"  Portfolio record : {record.id}")
    print(f"  Strategy DB copy (with updated pm_status) → {sc.PM_STRATEGY_DB_SHOWCASE}")
    print(f"  Portfolio DB (the allocation)             → {sc.PM_PORTFOLIO_DB_SHOWCASE}")
    print("\n  Try also:")
    print("      --personality defensive | aggressive   (different risk appetite)")
    print("      --committee                            (3 PMs vote on one book)")
    print("      --synthetic                            (degraded book → see flag/retire)")
    print("      --mode active                          (LLM-driven, needs OPENAI_API_KEY)")


if __name__ == "__main__":
    main()
