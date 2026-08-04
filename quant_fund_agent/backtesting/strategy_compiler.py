"""Deterministic strategy compiler: factor book → risk-managed portfolio.

Design record: ``docs/research-evolution/FACTOR_TO_STRATEGY_DESIGN.md`` (2026-08-03).
The factor book is the research product; this module is the fixed, LLM-free
conversion into tradable positions:

    prediction (blended RF+GBM over the book's signals)
      → cross-sectional z-score (winsorized)
      → EWMA time-smoothing               (turnover: trade the horizon, not the noise)
      → optional sector demean            (kill unintended sector bets)
      → risk scaling 1/σ_name             (equalize name-level risk)
      → optional beta neutralisation      (kill the unintended market bet)
      → cross-sectional demean + gross normalisation   (dollar-neutral LS leg)
      → net-exposure blend with the equal-weight long book   (persona choice)
      → max-positions concentration with entry/exit hysteresis (customer books)
      → no-trade band                     (suppress sub-threshold rebalancing)
      → volatility targeting with leverage cap

Personas parameterize ONLY the theme filter (which factors enter the model)
and the risk layer — never the alpha pipeline itself (``personas.yaml``).

Accounting matches ``scripts/backtest_combined_book.py``: 1-bar forward
mark-to-market with ±50% clip, net = pnl − COST_RATE·turnover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COST_RATE = 5e-4          # per unit of one-sided turnover (matches book backtest)
VOL_FLOOR = 1e-4          # daily per-name vol floor for risk scaling
_REPO = Path(__file__).resolve().parents[2]
PERSONAS_PATH = _REPO / "personas.yaml"

# Blended combiner menu.  "gbm" is genuine gradient boosting (lightgbm);
# "rf_gbm" is the RF-dominant blend ("Random Forest mit einem bisschen
# Gradient Boosting", decision 2026-08-03).
MODEL_BLENDS: dict[str, dict[str, float]] = {
    "gbm": {"lightgbm": 1.0},
    "rf": {"random_forest": 1.0},
    "rf_gbm": {"random_forest": 0.6, "lightgbm": 0.4},
}


@dataclass(frozen=True)
class RiskParams:
    """The persona-controlled risk layer (everything after the prediction)."""

    vol_target: float = 0.10        # annualised portfolio vol target
    net_exposure: float = 0.0       # 0 = market-neutral, 0.3 = 30% net long
    max_leverage: float = 3.0       # cap on the vol-targeting multiplier
    halflife: int = 6               # EWMA smoothing of the score (bars ≈ horizon)
    band: float = 0.10              # no-trade band, fraction of a typical position
    max_positions: int | None = None  # None = full breadth (research/automated)
    exit_buffer: float = 1.5        # leave the book only below rank N·buffer
    vol_halflife: int = 20          # per-name EWMA vol halflife (risk scaling)
    beta_neutral: bool = True
    sector_neutral: bool = True


@dataclass(frozen=True)
class Persona:
    key: str
    description: str = ""
    # theme filter (same semantics as scripts/simulate_user_strategies.select_factors)
    categories: tuple[str, ...] | None = None
    keywords: tuple[str, ...] = ()
    prefer_fundamental: bool = False
    diversify: bool = False
    max_factors: int = 24
    risk: RiskParams = field(default_factory=RiskParams)

    def theme(self) -> dict[str, Any]:
        """The dict `select_factors` expects."""
        return {"key": self.key,
                "categories": list(self.categories) if self.categories else None,
                "keywords": list(self.keywords),
                "prefer_fundamental": self.prefer_fundamental,
                "diversify": self.diversify,
                "max_factors": self.max_factors}


def load_personas(path: str | Path | None = None) -> list[Persona]:
    import yaml

    raw = yaml.safe_load(Path(path or PERSONAS_PATH).read_text())
    out = []
    for row in raw["personas"]:
        risk = RiskParams(**(row.get("risk") or {}))
        out.append(Persona(
            key=row["key"], description=row.get("description", ""),
            categories=tuple(row["categories"]) if row.get("categories") else None,
            keywords=tuple(row.get("keywords") or ()),
            prefer_fundamental=bool(row.get("prefer_fundamental", False)),
            diversify=bool(row.get("diversify", False)),
            max_factors=int(row.get("max_factors", 24)),
            risk=risk))
    return out


# ── prediction blending ──────────────────────────────────────────────────────

def cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Per-bar cross-sectional z-score, winsorized to ±clip."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def blend_predictions(preds: Mapping[str, pd.DataFrame],
                      weights: Mapping[str, float]) -> pd.DataFrame:
    """Weighted sum of per-model predictions, each z-scored per bar first so
    heterogeneous output scales (RF vs GBM) contribute by rank, not variance."""
    total = sum(weights[m] for m in preds)
    acc = None
    for name, p in preds.items():
        z = cs_zscore(p) * (weights[name] / total)
        acc = z if acc is None else acc.add(z, fill_value=0.0)
    return acc


# ── position pipeline ────────────────────────────────────────────────────────

def _sector_demean(s: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    lab = labels.reindex(s.columns).fillna("_none")
    means = s.T.groupby(lab).transform("mean").T
    return s - means


def _rolling_beta(r: pd.DataFrame, halflife: int = 63,
                  min_periods: int = 20) -> pd.DataFrame:
    rm = r.mean(axis=1)
    m_r = r.ewm(halflife=halflife, min_periods=min_periods).mean()
    m_m = rm.ewm(halflife=halflife, min_periods=min_periods).mean()
    cov = r.mul(rm, axis=0).ewm(halflife=halflife, min_periods=min_periods).mean() \
        - m_r.mul(m_m, axis=0)
    var = (rm * rm).ewm(halflife=halflife, min_periods=min_periods).mean() - m_m * m_m
    return cov.div(var.clip(lower=1e-10), axis=0)


def _band_and_concentrate(target: pd.DataFrame, risk: RiskParams,
                          w_init: np.ndarray | None = None) -> pd.DataFrame:
    """Sequential pass: max-positions hysteresis + no-trade band.

    Membership hysteresis: a name ENTERS the book only when it ranks inside the
    top ``max_positions`` by |target|; a held name LEAVES only when it drops
    below rank ``max_positions·exit_buffer`` (or its target dies).  Without the
    buffer a concentrated customer book churns names daily at the rank edge.
    """
    W = np.nan_to_num(target.to_numpy(dtype=float))
    T, N = W.shape
    held = np.zeros(N) if w_init is None else np.asarray(w_init, dtype=float).copy()
    out = np.empty_like(W)
    n_max = risk.max_positions
    exit_rank = int(round(n_max * risk.exit_buffer)) if n_max else None
    for t in range(T):
        wt = W[t].copy()
        alive = wt != 0.0
        if n_max:
            order = np.argsort(-np.abs(wt))            # rank 0 = strongest
            rank = np.empty(N, dtype=int)
            rank[order] = np.arange(N)
            entering = (rank < n_max) & alive
            staying = (held != 0.0) & (rank < exit_rank) & alive
            keep = entering | staying
            wt[~keep] = 0.0
            g = np.abs(wt).sum()
            if g > 0:
                wt *= np.abs(W[t]).sum() / g           # restore pre-cut gross
        n_active = max(int((wt != 0).sum()), n_max or 0, 1)
        tau = risk.band / n_active
        dh = wt - held
        trade = np.abs(dh) > tau
        held = np.where(trade, wt, held)
        # a name the pipeline no longer wants (invalid, or cut from the
        # concentrated book) is closed outright — the band only applies to
        # sizing changes of names still wanted
        held[wt == 0.0] = 0.0
        out[t] = held
    return pd.DataFrame(out, index=target.index, columns=target.columns)


def compile_positions(pred: pd.DataFrame, close: pd.DataFrame,
                      risk: RiskParams,
                      sector_labels: pd.Series | None = None,
                      w_init: np.ndarray | None = None) -> pd.DataFrame:
    """Full pipeline: prediction frame → final (vol-targeted) weight frame."""
    valid = pred.notna() & close.notna()
    r = close.pct_change().clip(-0.5, 0.5)

    z = cs_zscore(pred)
    s = z.ewm(halflife=risk.halflife, min_periods=1).mean().where(valid)
    if risk.sector_neutral and sector_labels is not None:
        s = _sector_demean(s, sector_labels)
    vol = r.ewm(halflife=risk.vol_halflife, min_periods=5).std()
    rs = s / vol.clip(lower=VOL_FLOOR)
    rs = rs.sub(rs.mean(axis=1), axis=0)                 # dollar-neutral LS leg
    if risk.beta_neutral:
        # project against the DEMEANED beta: it lives in the zero-sum subspace,
        # so the projection removes the market bet without breaking dollar
        # neutrality (projecting raw beta and demeaning afterwards would shift
        # along the ones-vector, whose beta-overlap reintroduces the bet).
        beta = _rolling_beta(r).where(valid)
        beta_d = beta.sub(beta.mean(axis=1), axis=0).fillna(0.0)
        num = (rs * beta_d).sum(axis=1)
        den = (beta_d * beta_d).sum(axis=1).clip(lower=1e-12)
        rs = rs - beta_d.mul(num / den, axis=0)
    gross = rs.abs().sum(axis=1).replace(0.0, np.nan)
    w_ls = rs.div(gross, axis=0)

    nu = float(risk.net_exposure)
    if nu:
        # Net exposure from STOCK PICKING, not an index sleeve (decision
        # 2026-08-03): the long leg overweights the model's best-ranked names,
        # the short leg shrinks to the worst ones.  Long gross (1+ν)/2, short
        # gross (1−ν)/2 → net = ν, gross = 1; ν=1 is a pure long-only picking
        # book.  Market beta then emerges from the picks themselves.
        score = rs
        pos = score.clip(lower=0.0)
        neg = (-score).clip(lower=0.0)
        g_long = (1.0 + nu) / 2.0
        g_short = (1.0 - nu) / 2.0
        w = (pos.div(pos.sum(axis=1).replace(0.0, np.nan), axis=0) * g_long
             - neg.div(neg.sum(axis=1).replace(0.0, np.nan), axis=0) * g_short)
    else:
        w = w_ls
    w = w.fillna(0.0)

    held = _band_and_concentrate(w, risk, w_init=w_init)

    # causal vol targeting: leverage at t uses base returns realised through t
    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)
    base = (held * fwd1).sum(axis=1)                     # base_t realised at t+1
    realised = base.ewm(halflife=20, min_periods=20).std().shift(1) * np.sqrt(252)
    lev = (risk.vol_target / realised).clip(upper=risk.max_leverage)
    lev = lev.fillna(1.0)
    return held.mul(lev, axis=0)


def strategy_returns(weights: pd.DataFrame, close: pd.DataFrame,
                     cost_rate: float = COST_RATE
                     ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(net, turnover, gross-pnl) under the shared book-backtest accounting."""
    fwd1 = (close.shift(-1) / close - 1.0).clip(-0.5, 0.5)
    pnl = (weights * fwd1).sum(axis=1)
    turnover = (weights - weights.shift(1)).abs().sum(axis=1)
    net = pnl - cost_rate * turnover
    return net.iloc[:-1], turnover.iloc[:-1], pnl.iloc[:-1]


def modal_sector_labels(sector_panel: pd.DataFrame) -> pd.Series:
    """Per-name modal sector label from the (mostly constant) sector field."""
    def _mode(col: pd.Series):
        vc = col.dropna()
        return vc.mode().iloc[0] if len(vc) else np.nan
    return sector_panel.apply(_mode, axis=0)
