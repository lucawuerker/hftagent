"""Position-construction primitives shared by research and deployment.

Two regimes turn a composite signal (time × ticker) into tradeable positions:

* **cross-sectional** (``strategy_backtester._signal_to_positions``) — rank the
  cross-section each bar and build a dollar-neutral long/short book.  The natural
  choice for a homogeneous universe (e.g. S&P 500 stocks traded against each
  other).
* **per-underlying** (this module) — treat each underlying independently: stand-
  ardise its signal over its *own* history, then go long / short / flat by a
  fixed boundary.  Directional (net market exposure), and needs no cross-section,
  so it suits a heterogeneous, data-rich universe (e.g. the LOBSTER ETFs).

``zscore_over_time`` and ``directional_positions`` are the exact primitives the
research-side comparison track uses (``comparison.vector_backtest``), factored
out here so "per-underlying threshold band" means byte-identical things in
research and in the deployed fund — no drift between what we measure and trade.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Defaults for the per-underlying regime (mirror the comparison-track defaults).
DEFAULT_POSITION_MODE = "threshold"        # "threshold" | "sign" | "continuous"
DEFAULT_POSITION_THRESHOLD = 1.0           # ±t (z units) for the threshold band
DEFAULT_ZSCORE_BASIS = "expanding"         # "expanding" | "full" | "rolling" | "none"
DEFAULT_ZSCORE_WINDOW = 500                # window for the "rolling" basis


def zscore_over_time(frame: pd.DataFrame, basis: str, window: int) -> pd.DataFrame:
    """Per-underlying (per-column) standardisation of a signal over time.

    ``expanding`` (default) is causal — only past+present inform each row, so it
    is leak-free for a live/OOS run.  ``full`` uses whole-sample stats (mild
    look-ahead; simplest), ``rolling`` a trailing window, ``none`` a pass-through.
    """
    if basis == "none":
        return frame
    if basis == "full":
        return (frame - frame.mean()) / frame.std().replace(0, np.nan)
    if basis == "rolling":
        m = frame.rolling(window, min_periods=max(2, window // 5)).mean()
        s = frame.rolling(window, min_periods=max(2, window // 5)).std()
        return (frame - m) / s.replace(0, np.nan)
    # expanding (default): causal.
    m = frame.expanding(min_periods=2).mean()
    s = frame.expanding(min_periods=2).std()
    return (frame - m) / s.replace(0, np.nan)


def directional_positions(
    z: pd.DataFrame, mode: str = DEFAULT_POSITION_MODE, threshold: float = DEFAULT_POSITION_THRESHOLD,
) -> pd.DataFrame:
    """Map a (z-scored) signal to a raw directional position per underlying.

    ``sign`` → ``±1``; ``continuous`` → ``z`` clipped to ``[-1, 1]``; ``threshold``
    (default) → ``+1`` above ``+threshold``, ``-1`` below ``-threshold``, else
    ``0``.  Magnitudes are ``≤ 1``; sizing into book weights is applied later.
    """
    if mode == "sign":
        return np.sign(z).fillna(0.0)
    if mode == "continuous":
        return z.clip(-1.0, 1.0).fillna(0.0)
    # threshold band (default): long above +t, short below −t, flat in between.
    pos = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    return pos.mask(z > threshold, 1.0).mask(z < -threshold, -1.0)


def per_underlying_positions(
    signal: pd.DataFrame,
    *,
    n_max_positions: int,
    holding_period: int = 1,
    mode: str = DEFAULT_POSITION_MODE,
    threshold: float = DEFAULT_POSITION_THRESHOLD,
    zscore_basis: str = DEFAULT_ZSCORE_BASIS,
    zscore_window: int = DEFAULT_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """Directional per-underlying book weights from a composite signal.

    Steps:
      1. hold the position between rebalances (``holding_period``);
      2. standardise each underlying's signal over its own history (``zscore_basis``);
      3. map to a raw directional position (long / flat / short) via the boundary;
      4. size each active position to a fixed ``1 / n_max_positions`` (equal
         weight), signed by the directional decision.

    Sizing is intentionally simple to start: a name that fires gets ``1/n``, so a
    single long signal commits only ``1/n`` of capital (the rest stays in cash),
    never the whole book.  Gross exposure is ``(# active names) / n_max_positions``
    — ``≤ 1`` until more than ``n_max_positions`` fire at once.

    NOTE: more sophisticated sizing (e.g. inverse-volatility weighting, or a hard
    cap on concurrent names) may replace the flat ``1/n`` here later; it is the
    only thing this function decides, so it is the single place to change.
    """
    # 1 — holding-period resampling (mirror _signal_to_positions step 1).
    if holding_period > 1:
        mask = np.arange(len(signal)) % holding_period == 0
        signal = signal.where(pd.Series(mask, index=signal.index), other=np.nan).ffill()

    # 2/3 — per-underlying z-score → directional position.
    z = zscore_over_time(signal, zscore_basis, zscore_window)
    raw = directional_positions(z, mode=mode, threshold=threshold)

    # 4 — equal-weight sizing: each active position = 1 / n_max_positions.
    n = max(1, int(n_max_positions))
    return raw / float(n)


# ── construction-regime resolution (default by data type + override) ─────────

VALID_CONSTRUCTIONS = ("cross_sectional", "per_underlying")


def default_position_construction(settings=None) -> str:
    """Provider-default regime: ``per_underlying`` for LOBSTER, else ``cross_sectional``.

    LOBSTER is a small, heterogeneous, data-rich universe (ETFs across sectors)
    where a directional per-name bet makes sense; the API providers serve larger,
    homogeneous equity universes traded cross-sectionally market-neutral.
    """
    try:
        from quant_fund_agent.config import get_settings

        settings = settings or get_settings()
        provider = str(getattr(settings.data, "provider", "")).lower()
    except Exception:  # noqa: BLE001 — never let config resolution break a backtest
        provider = ""
    return "per_underlying" if provider == "lobster" else "cross_sectional"


def resolve_position_construction(override: str | None = None, settings=None) -> str:
    """Resolve the regime: explicit override > ``QF_POSITION_CONSTRUCTION`` > provider default.

    ``override`` / the env var accept ``"auto"`` (use the provider default),
    ``"cross_sectional"`` or ``"per_underlying"``; anything else falls back to the
    provider default.
    """
    val = (override or os.getenv("QF_POSITION_CONSTRUCTION") or "auto").strip().lower()
    if val in VALID_CONSTRUCTIONS:
        return val
    return default_position_construction(settings)
