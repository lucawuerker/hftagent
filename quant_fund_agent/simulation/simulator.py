"""The walk-forward backtest driver.

``run_backtest(config)`` steps the fund through time on a weekly grid, driving
the agents **only** through the :mod:`quant_fund_agent.pipeline` seam (so the
same code path runs in live trading), and trading the resulting book on unseen
data with the :mod:`~quant_fund_agent.simulation.execution` layer.

Per grid point (period):
  1. (if due) **research meeting** — Factor Researcher invents factors (as-of cutoff).
  2. (if due) **strategy meeting** — Selector→Architect→Statistician build & validate
     strategies (as-of cutoff); approved ones are persisted (frozen) into the book.
  3. (if due) **PM meeting** — the PM (committee) reallocates capital.
  4. **trade** the live window ``[cutoff, next)`` with the current allocation,
     netting positions into one fund book (or per-pod), net of costs.
  5. mark each strategy's live PnL (for the PM's monitoring next meeting).

Strategies are **frozen on approval**: their fitted artifacts are never refit;
the PM monitors and may retire them, and only new research adds strategies.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from quant_fund_agent.simulation.config import BacktestConfig
from quant_fund_agent.simulation.results import BacktestResults

log = logging.getLogger("simulation.simulator")


def run_backtest(config: BacktestConfig) -> BacktestResults:
    """Run a walk-forward backtest and return the accumulated results."""
    # ── env must be set BEFORE the modeling service / panel are imported,
    #    because DATA_DIR is read at module import and the universe cap at call.
    os.environ["DATA_DIR"] = config.data_dir
    if config.n_tickers is not None:
        os.environ["ARCHITECT_N_TICKERS"] = str(config.n_tickers)

    import numpy as np

    np.random.seed(config.seed)

    from quant_fund_agent import pipeline
    from quant_fund_agent.backtesting.strategy_backtester import backtest_strategy
    from quant_fund_agent.databases import PortfolioDatabase, StrategyDatabase
    from quant_fund_agent.schemas import PMStatus
    from quant_fund_agent.simulation.clock import TradingClock
    from quant_fund_agent.simulation.signals import SignalCache

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── databases scoped to THIS run (don't clobber the live book) ──
    if config.fresh:
        strategy_db = StrategyDatabase(returns_dir=config.returns_dir)
        portfolio_db = PortfolioDatabase()
    else:
        strategy_db = StrategyDatabase(returns_dir=config.returns_dir)
        strategy_db.load_from_json(config.strategy_db_path)
        portfolio_db = PortfolioDatabase()
        portfolio_db.load_from_json(config.portfolio_db_path)

    signals = SignalCache()
    clock = TradingClock(signals.index, config)
    results = BacktestResults(config)

    periods = clock.periods()
    log.info("Walk-forward: %d periods, %s → %s (%s book)",
             len(periods), config.start, config.end, config.execution_model.value)
    if not periods:
        log.warning("No periods to simulate — check start/end/warmup vs. data span.")
        return results

    live_returns_accum: dict[str, list[pd.Series]] = {}
    first_research_done = False

    for p in periods:
        meeting: dict = {
            "cutoff": p.cutoff.isoformat(),
            "week": p.week_tag,
            "research": p.research_due,
            "strategy": p.strategy_due,
            "pm": p.pm_due,
        }

        # 1 ── research meeting ────────────────────────────────────────
        if p.research_due:
            try:
                reset = config.fresh and not first_research_done
                rs = pipeline.run_research_session(
                    session_id=p.week_tag,
                    cutoff_date=p.cutoff_date,
                    n_tickers=config.n_tickers or 15,
                    reset=reset,
                )
                first_research_done = True
                meeting["kept_factor_ids"] = rs.get("kept_factor_ids", [])
            except Exception as e:  # noqa: BLE001 — one bad meeting must not abort the run
                log.warning("[%s] research meeting failed: %s", p.week_tag, e)
                meeting["research_error"] = str(e)

        # 2 ── strategy meeting ────────────────────────────────────────
        if p.strategy_due:
            k = config.initial_strategies if p.is_first_strategy_meeting \
                else config.n_strategies_per_meeting
            approved: list[str] = []
            for _ in range(k):
                try:
                    res = pipeline.run_strategy_pipeline(
                        max_iterations=config.max_iterations,
                        oos_ratio=config.oos_ratio,
                        target_horizon=config.target_horizon,
                        cutoff_date=p.cutoff_date,
                    )
                    rec = pipeline.persist_approved_strategy(strategy_db, res)
                    if rec is not None:
                        approved.append(rec.id)
                except Exception as e:  # noqa: BLE001
                    log.warning("[%s] strategy pipeline failed: %s", p.week_tag, e)
            meeting["strategies_built"] = k
            meeting["strategies_approved"] = approved

        # 3 ── PM meeting ──────────────────────────────────────────────
        record = None
        if p.pm_due and strategy_db.list_strategies():
            try:
                record = pipeline.run_pm_rebalance(
                    strategy_db, portfolio_db,
                    personalities=config.pm_personalities,
                    mode=config.pm_mode,
                    committee=config.committee,
                    voting_method=config.voting_method,
                    pm_name=config.pm_name,
                    construction_method=config.construction_method,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] PM rebalance failed: %s", p.week_tag, e)

        if record is None:
            record = portfolio_db.latest(config.pm_name)  # carry forward last allocation

        weights = _allocation_weights(record)
        if record is not None and p.pm_due:
            results.log_portfolio({
                "cutoff": p.cutoff.isoformat(),
                "pm_name": record.pm_name,
                "construction_method": record.construction_method.value,
                "allocations": {a.strategy_id: a.weight for a in record.allocations if a.enabled},
                "retired": list(record.retired_strategy_ids),
                "flagged": list(record.flagged_strategy_ids),
                "expected_metrics": record.expected_metrics,
            })
        meeting["allocation"] = weights

        # 4 ── trade the live window ───────────────────────────────────
        exec_result = _trade_window(
            weights, strategy_db, signals, p.live_start, p.live_end,
            backtest_strategy, config, PMStatus,
        )
        if exec_result is not None:
            results.add_window(exec_result)
            meeting["live_bars"] = int(len(exec_result.fund_returns))
            meeting["window_return"] = round(float(exec_result.fund_returns.sum()), 6)

            # 5 ── mark each strategy's live PnL for PM monitoring next time ──
            _mark_live(exec_result, strategy_db, live_returns_accum)

        results.log_meeting(meeting)

        # periodic flush
        if p.index > 0 and p.index % config.persist_every_n_steps == 0:
            pipeline.save_dbs(strategy_db, portfolio_db,
                              config.strategy_db_path, config.portfolio_db_path)

    # ── finalise ────────────────────────────────────────────────────────
    pipeline.save_dbs(strategy_db, portfolio_db,
                      config.strategy_db_path, config.portfolio_db_path)
    results.finalize()
    results.save()
    log.info("\n%s", results.summary())
    return results


# ── helpers ──────────────────────────────────────────────────────────────

def _allocation_weights(record) -> dict[str, float]:
    if record is None:
        return {}
    return {
        a.strategy_id: float(a.weight)
        for a in record.allocations
        if a.enabled and abs(a.weight) > 1e-9
    }


def _trade_window(
    weights, strategy_db, signals, live_start, live_end,
    backtest_strategy, config, PMStatus,
):
    """Reconstruct each live strategy's positions on the window and execute."""
    from quant_fund_agent.simulation.execution import run_execution

    if not weights:
        return None

    panel_live = signals.panel_window(live_start, live_end)
    if not panel_live or panel_live.get("close") is None or panel_live["close"].empty:
        return None

    positions: dict[str, pd.DataFrame] = {}
    for sid in list(weights.keys()):
        rec = strategy_db.get_strategy(sid)
        if rec is None or rec.pm_status == PMStatus.RETIRED:
            weights.pop(sid, None)
            continue
        try:
            from quant_fund_agent import pipeline

            strat = pipeline.strategy_from_record(rec)
            fids = list(strat.factor_ids) or list((rec.model_params or {}).get("weights", {}).keys())
            signals_live = signals.signals_window(fids, live_start, live_end)
            if not signals_live:
                weights.pop(sid, None)
                continue
            res = backtest_strategy(strat, signals_live, panel_live)
            positions[sid] = res.positions
        except Exception as e:  # noqa: BLE001 — a single broken strategy shouldn't halt trading
            log.warning("trade: strategy %s failed to run on live window: %s", sid, e)
            weights.pop(sid, None)

    if not positions:
        return None

    return run_execution(positions, weights, panel_live, config)


def _mark_live(exec_result, strategy_db, live_returns_accum) -> None:
    """Append live PnL + refresh each strategy's live_metrics for the PM."""
    for sid, net_s in exec_result.strategy_returns.items():
        if net_s is None or len(net_s) == 0:
            continue
        try:
            strategy_db.append_returns(sid, net_s)
        except KeyError:
            continue
        live_returns_accum.setdefault(sid, []).append(net_s)
        rec = strategy_db.get_strategy(sid)
        if rec is not None:
            full_live = pd.concat(live_returns_accum[sid])
            full_live = full_live[~full_live.index.duplicated(keep="last")].sort_index()
            rec.live_metrics = _metrics_from_returns(full_live)


def _metrics_from_returns(returns: pd.Series):
    """Build a StrategyBacktestMetrics from a per-bar live return series."""
    from quant_fund_agent.schemas import StrategyBacktestMetrics
    from quant_fund_agent.simulation.results import fund_metrics

    m = fund_metrics(returns)
    return StrategyBacktestMetrics(
        annualised_return=m.get("annualised_return"),
        annualised_volatility=m.get("annualised_volatility"),
        sharpe_ratio=m.get("sharpe_ratio"),
        sortino_ratio=m.get("sortino_ratio"),
        calmar_ratio=m.get("calmar_ratio"),
        max_drawdown=m.get("max_drawdown"),
        max_drawdown_duration_bars=m.get("max_drawdown_duration_bars"),
        hit_rate=m.get("hit_rate"),
    )
