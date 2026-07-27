"""Prompts used by the Factor Researcher Agent.

Kept in their own module for two reasons:
  1. They are long and would otherwise bury the graph wiring.
  2. They will be iterated on independently of the graph logic
     (the LangGraph plumbing rarely changes; the prompt wording
     changes constantly during prompt-engineering).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# DATA CONTEXT — built dynamically from the field set this run can supply.
#
# The Factor Researcher must only invent factors the configured data feed can
# actually serve, so the data-context prose is assembled by ``build_data_context``
# from an ``allowed_fields`` set (provider capabilities + LOBSTER order-book level
# + the fundamentals opt-out; see ``quant_fund_agent.data.usable_fields``).  Each
# field below is a pre-formatted entry; only the allowed ones are emitted, and a
# whole section (microstructure / fundamentals / per-level book) is dropped when
# none of its fields are available.
# ---------------------------------------------------------------------------

# Price / OHLCV view (the ``standard`` tier — mid-derived on LOBSTER).
_PRICE_ENTRIES: dict[str, str] = {
    "open": "    open      : start-of-bar price (mid-price on LOBSTER)",
    "high": "    high      : intra-bar high (max(mid, midEnd) on LOBSTER)",
    "low": "    low       : intra-bar low (min(mid, midEnd) on LOBSTER)",
    "close": "    close     : end-of-bar price (mid-price on LOBSTER)",
    "volume": "    volume    : traded volume in the bar (|signed trade| on LOBSTER)",
}

# Raw LOBSTER microstructure fields (signed quantities use +buy / -sell).
_MICRO_ENTRIES: dict[str, str] = {
    "trade": (
        "    trade     : signed traded volume in the bar (shares; +buyer-initiated,\n"
        "                -seller-initiated). ``volume`` above is its absolute value."
    ),
    "orderFlow": (
        "    orderFlow : signed limit-order-flow into the top of book (shares).\n"
        "                Positive = net additions to the bid side / cancellations\n"
        "                on the ask, negative = the mirror."
    ),
    "hidden": "    hidden    : hidden traded volume in the bar (shares).",
    "auction": (
        "    auction   : auction-print volume in the bar (shares; usually zero\n"
        "                outside the open / close auction windows)."
    ),
    "spread": "    spread    : end-of-bar quoted bid-ask spread (price units).",
    "effSpread": (
        "    effSpread : volume-weighted effective spread in the bar — captures\n"
        "                price impact of the trades that printed."
    ),
    "lobImb": (
        "    lobImb    : top-of-book limit-order-book imbalance,\n"
        "                (bid_size - ask_size) / (bid_size + ask_size), in [-1, +1]."
    ),
    "effLobImb": (
        "    effLobImb : effective LOB imbalance weighted by depth across the\n"
        "                visible book — a sturdier book-pressure signal."
    ),
    "trdLiq": (
        "    trdLiq    : trade-side liquidity proxy (size traded per unit price\n"
        "                move during the bar; higher = more liquid)."
    ),
    "ofLiq": (
        "    ofLiq     : order-flow liquidity proxy (size posted per unit price\n"
        "                move; higher = denser book)."
    ),
    "depth": "    depth     : average top-of-book depth (best bid + best ask sizes).",
    "nbEvents": "    nbEvents  : number of LOB events (any update) in the bar.",
    "nbHidden": "    nbHidden  : number of hidden trade prints in the bar.",
    "nbTrades": "    nbTrades  : number of visible trade prints in the bar.",
}

_LEVEL_BOOK_NOTE = """\
Per-level order-book fields (when the feed carries depth beyond the top of
book) are exposed as ``askPrice{i}`` / ``askDepth{i}`` / ``bidPrice{i}`` /
``bidDepth{i}`` for each visible level ``i`` (1 = best), e.g. ``data["bidDepth1"]``.
Deeper levels are NaN when the pull was shallower — guard with ``.fillna``."""

_FUNDAMENTAL_ENTRIES: dict[str, str] = {
    "sector": '    sector    : GICS-style sector label (text, e.g. "Technology"). Static.',
    "industry": "    industry  : finer industry label (text). Static.",
    "marketCap": "    marketCap : market capitalization in USD (daily where the feed\n"
                 "                supplies it, otherwise per fiscal quarter).",
    "peRatio": "    peRatio   : price / earnings (float; negative for loss-makers).",
    "pbRatio": "    pbRatio   : price / book.",
    "psRatio": "    psRatio   : price / sales.",
    "roe": "    roe       : return on equity.",
    "roic": "    roic      : return on invested capital.",
    "debtToEquity": "    debtToEquity : leverage ratio.",
    "currentRatio": "    currentRatio : liquidity ratio.",
    "grossMargin": "    grossMargin : gross profitability margin (fraction).",
    "netMargin": "    netMargin   : net profitability margin (fraction).",
    "revenue": "    revenue   : quarterly revenue (USD).",
    "eps": "    eps       : reported EPS (USD).",
    "freeCashFlow": "    freeCashFlow : free cash flow in USD (operating cash flow − capex).",
    "epsEstimate": "    epsEstimate    : analyst EPS consensus for the latest quarter.",
    "revenueEstimate": "    revenueEstimate: analyst revenue consensus for the latest quarter.",
    "epsSurprise": "    epsSurprise : reported EPS − estimate (post-earnings-drift signal).",
}


def _archive_entries() -> dict[str, str]:
    """Prose lines for the premium-archive fields, grouped and machine-generated.

    The premium FMP archive serves ~130 fundamental fields; hand-writing a line
    for each would guarantee drift from :data:`data.fields.ARCHIVE_FIELD_SPECS`.
    Instead the descriptions live *with* the field definitions and are rendered
    here, grouped so a long list still reads as an ordered menu rather than a
    wall of names.  Hand-written entries above take precedence for the fields
    they already cover (they carry extra nuance worth keeping).
    """
    from quant_fund_agent.data.fields import ARCHIVE_FIELD_SPECS, FIELD_GROUPS

    width = max((len(s.name) for s in ARCHIVE_FIELD_SPECS), default=12)
    out: dict[str, str] = {}
    for group in FIELD_GROUPS:
        for spec in ARCHIVE_FIELD_SPECS:
            if spec.group == group and spec.name not in _FUNDAMENTAL_ENTRIES:
                out[spec.name] = f"    {spec.name:<{width}}: {spec.desc}"
    return out


# Hand-written entries first (they read better for the core fields), then every
# remaining archive field.  ``_section`` preserves this declaration order, so the
# gated list stays grouped.
_FUNDAMENTAL_ENTRIES.update(_archive_entries())

_FUNDAMENTAL_LOOKAHEAD = """\
LOOK-AHEAD — read carefully.  These fields are **already point-in-time**:
each value is stamped at its *availability date* (the filing / report date,
or fiscal-period-end + a reporting lag) and forward-filled, so reading
``data["peRatio"]`` at date t only ever sees what was public by t.  You do
NOT need to (and must not) shift them yourself.  They are **quarterly step
functions**: ``NaN`` before a name's first report and after a long staleness
gap, and otherwise constant between reports — so be defensive (``.fillna``,
``df.where(...)``) and prefer cross-sectional ops (``rank``) and slow changes
(quarter-over-quarter ``delta``) over fast time-series ops."""


def _section(allowed, entries: dict[str, str]) -> list[str]:
    """The pre-formatted entry lines for the allowed fields, in declared order."""
    return [text for name, text in entries.items() if allowed is None or name in allowed]


# Hard guardrail used by the codegen validator and quoted in the prompts: a
# factor's prediction horizon must be a positive int no larger than this many
# bars.  It only catches pathological values (e.g. a horizon of 1e9); the
# *runtime* further clamps the effective horizon to the loaded panel length.
MAX_PREDICTION_HORIZON: int = 2000


def _bar_size_phrase(seconds_per_bar: float | None) -> str | None:
    """A human bar-size noun phrase (without the trailing "bars").

    ``10`` → ``"10-second"``, ``60`` → ``"1-minute"``, ``86400`` → ``"daily"``.
    Returns ``None`` when the bar size is unknown so callers fall back to
    feed-agnostic wording (we never assume a default such as 10s).
    """
    if not seconds_per_bar or seconds_per_bar <= 0:
        return None
    s = float(seconds_per_bar)

    def _num(x: float) -> str:
        return str(int(round(x))) if abs(x - round(x)) < 1e-6 else f"{x:.1f}"

    if s < 60:
        return f"{_num(s)}-second"
    if s < 3600:
        return f"{_num(s / 60)}-minute"
    if s < 86400:
        return f"{_num(s / 3600)}-hour"
    days = s / 86400
    return "daily" if abs(days - 1.0) < 1e-6 else f"{_num(days)}-day"


def _horizon_contract(
    seconds_per_bar: float | None,
    fixed_prediction_horizon: int | None = None,
) -> str:
    """The PREDICTION HORIZON contract appended to every data context.

    Tells the LLM the bar size (so it can reason in wall-clock time), that the
    horizon is expressed in *bars*, and the legal range — shared by the
    brainstorm output schema and the codegen ``prediction_horizon`` requirement.
    """
    phrase = _bar_size_phrase(seconds_per_bar)
    if fixed_prediction_horizon is not None:
        h = int(fixed_prediction_horizon)
        if phrase is not None:
            unit = f"each bar spans **{phrase.replace('-', ' ')}** of wall-clock time"
            example = f"so this run is fixed at ~{h} {phrase} bars ahead"
        else:
            unit = "the bar size is set by the configured feed"
            example = f"so this run is fixed at {h} bars ahead"
        return (
            "PREDICTION HORIZON\n"
            "------------------\n"
            f"This run uses ONE fixed forecast horizon for every factor: "
            f"``prediction_horizon = {h}`` bars ({unit}; {example}).\n"
            f"Do not choose a separate horizon for the idea.  Every brainstormed "
            f"idea and every generated factor class must declare exactly "
            f"``prediction_horizon = {h}``; set ``suggested_horizons`` to "
            f"``[{h}]`` or omit it.  Valid range: "
            f"1 ≤ prediction_horizon ≤ {MAX_PREDICTION_HORIZON} bars."
        )

    if phrase is not None:
        unit = f"each bar spans **{phrase.replace('-', ' ')}** of wall-clock time"
        example = (
            f"a ``prediction_horizon`` of 6 means ~6 {phrase} bars ahead"
        )
    else:
        unit = "the bar size is set by the configured feed"
        example = "a ``prediction_horizon`` of 6 means 6 bars ahead"
    return (
        "PREDICTION HORIZON\n"
        "------------------\n"
        f"Every factor must declare the horizon at which its edge is expected to\n"
        f"materialise, as an integer number of *bars* ({unit}).  This is the\n"
        f"forward offset its signal predicts — {example}.  Pick the horizon your\n"
        f"idea is actually about: a fast reversal / book-pressure signal is a few\n"
        f"bars; a slower trend or value signal is many.  Optionally also list a\n"
        f"few alternative ``suggested_horizons`` worth measuring.  Valid range:\n"
        f"1 ≤ prediction_horizon ≤ {MAX_PREDICTION_HORIZON} bars."
    )


def build_data_context(
    allowed_fields=None,
    seconds_per_bar: float | None = None,
    fixed_prediction_horizon: int | None = None,
) -> str:
    """Assemble the DATA CONTEXT prose for the fields this run can supply.

    ``allowed_fields`` is the set the configured data feed actually serves (see
    :func:`quant_fund_agent.data.usable_fields`).  ``None`` keeps every field
    (full LOBSTER + fundamentals) — the historical, un-gated behaviour.  Only
    allowed fields are listed and empty sections are dropped, so the researcher
    is never told about data it cannot use.

    ``seconds_per_bar`` is the data feed's bar size (inferred from the loaded
    panel index, see :func:`quant_fund_agent.data.frequency`); it drives the
    bar-size wording and the PREDICTION HORIZON contract.  ``None`` → feed-
    agnostic wording (no assumed default).

    ``fixed_prediction_horizon`` pins every generated factor to one run-level
    horizon instead of letting the researcher choose per idea.
    """
    allowed = set(allowed_fields) if allowed_fields is not None else None
    bar_phrase = _bar_size_phrase(seconds_per_bar)

    price_lines = _section(allowed, _PRICE_ENTRIES)
    micro_lines = _section(allowed, _MICRO_ENTRIES)
    fund_lines = _section(allowed, _FUNDAMENTAL_ENTRIES)
    has_levels = allowed is None or any(
        name.startswith(("askPrice", "askDepth", "bidPrice", "bidDepth"))
        for name in allowed
    )
    is_lobster = bool(micro_lines) or has_levels

    parts: list[str] = []
    if is_lobster:
        bar_note = (f"{bar_phrase} bars on the sampled feed" if bar_phrase
                    else "a sampled order-book feed")
        parts.append(
            f"You are working with LOBSTER-derived microstructure bars ({bar_note}).\n"
            "When you write a factor, the ``data`` dict passed to ``calc`` exposes the\n"
            "fields below as ``pd.DataFrame`` (index = bar timestamps, columns =\n"
            "tickers).  Every field below is available — pick whichever ones your idea\n"
            "actually needs; do NOT use any field that is not listed (the factor would\n"
            "be rejected as out-of-scope)."
        )
    else:
        feed_note = (f"the configured market-data feed ({bar_phrase} bars)"
                     if bar_phrase else "the configured market-data feed")
        parts.append(
            f"You are working with {feed_note}.  When you write a factor, the ``data``\n"
            "dict passed to ``calc`` exposes the fields below as ``pd.DataFrame``\n"
            "(index = bar timestamps, columns = tickers).  Every field below is\n"
            "available — pick whichever ones your idea actually needs; do NOT use any\n"
            "field that is not listed (the factor would be rejected as out-of-scope)."
        )

    if price_lines:
        parts.append("Price / OHLCV view:\n\n" + "\n".join(price_lines))
    if micro_lines:
        parts.append(
            "Microstructure fields (signed quantities use +buy / -sell convention):"
            "\n\n" + "\n".join(micro_lines)
        )
    if has_levels:
        parts.append(_LEVEL_BOOK_NOTE)

    notes = [
        "Notes:",
        "- All fields are aligned on the same DatetimeIndex and the same ticker",
        "  columns, so cross-field arithmetic is safe.",
        "- Do NOT pass extra keyword arguments to the helper operators below —",
        "  they take positional arguments only.",
    ]
    if is_lobster:
        notes.insert(1,
            "- Many fields are sparse (lots of NaN / zero on quiet bars).  Be\n"
            "  defensive on rolling ops: use ``.fillna``, ``.replace(0, np.nan)``,\n"
            "  ``df.where(...)``, etc.")
    parts.append("\n".join(notes))

    if fund_lines:
        # Only mention the fundamentals look-ahead caveat when those fields are in play.
        parts.append(
            "Fundamental / estimate / event fields (point-in-time, quarterly):\n\n"
            + "\n".join(fund_lines)
        )
        parts.append(_FUNDAMENTAL_LOOKAHEAD)

    parts.append(_horizon_contract(seconds_per_bar, fixed_prediction_horizon))

    return "\n\n".join(parts) + "\n"


# Full, un-gated context (every LOBSTER field + fundamentals).  Kept as a module
# constant for back-compat with importers that don't thread an allowed-field set.
DATA_CONTEXT = build_data_context(None)


# ---------------------------------------------------------------------------
# Operator reference — kept in the prompt verbatim because the LLM
# would otherwise invent plausible-but-wrong keyword arguments like
# ``ts_sum(x, window=5)`` or ``delta(x, min_periods=…)``.  Every signature
# below is taken directly from ``quant_fund_agent/factors/ops.py``.
# ---------------------------------------------------------------------------

OPERATOR_REFERENCE = """\
OPERATOR REFERENCE & IMPORT CONTRACT
------------------------------------
You are an LLM researcher, not a formula-recombiner.  Read this first: the
``ops`` library below is a *convenience shelf* of common primitives — it is
NOT the limit of what you may compute, and most interesting factors will need
more than it offers.  You are strongly encouraged to write your OWN vectorised,
causal helper functions to implement the *actual* mathematical machinery a
paper describes — truncated path signatures, Hawkes-process intensities,
spectral / wavelet / entropy features, rolling eigen-decompositions,
Ornstein-Uhlenbeck half-lives, variance ratios, robust regressions, and so on.
If a paper's mechanism is richer than rank/mean/delta, implement the richer
mechanism; do NOT water it down to fit the shelf.

THREE PLACES COMPUTATION COMES FROM - use whichever fits, mix them freely:

1. YOUR OWN HELPER FUNCTIONS - preferred for anything non-trivial.
   Define module-level ``def _my_transform(df, ...):`` helpers in the file and
   call them from ``calc``.  Build them from numpy / scipy / statsmodels /
   pandas.  This is where signature transforms, Hawkes intensities, spectral
   features, etc. live.  They MUST be vectorised over the panel, causal
   (trailing windows and positive lags only), and deterministic.

2. ``quant_fund_agent.factors.ops`` - a convenience library of WorldQuant-style
   primitives.  Reach for them when they happen to fit; skip them when they
   don't.  They are FREE FUNCTIONS, take POSITIONAL arguments ONLY (never
   ``window=``, ``n=``, ``min_periods=``, ``axis=``), and act on whole
   DataFrames.  The names below are the ONLY names that exist in ``ops``:

   Data helpers (consume the full ``data`` dict):
       returns(data)            close.pct_change(); uses data["returns"] if present
       vwap(data)               uses data["vwap"] or (H+L+C)/3 fallback

   Cross-sectional (per-row, no window):
       rank(df)                 percentile rank in [0, 1]

   Time-series (n = positive int, lookback in bars):
       delta(df, n)             df[t] - df[t-n]
       delay(df, n)             df.shift(n)
       ts_sum(df, n)
       ts_mean(df, n)
       stddev(df, n)
       ts_min(df, n)
       ts_max(df, n)
       ts_argmax(df, n)         1-indexed position of max in last n bars
       ts_argmin(df, n)
       ts_rank(df, n)           percentile rank of the latest value in last n bars
       product(df, n)           rolling product
       decay_linear(df, n)      linear-weighted MA, newest weight = n
       adv(volume_df, n)        n-bar average of volume

   Pairwise / math:
       correlation(x, y, n)     rolling Pearson corr between x and y over n
       covariance(x, y, n)
       signed_power(df, a)      sign(df) * |df|^a
       power(df, a)             df ** a
       log(df)                  natural log; zeros are masked to NaN
       abs_(df)
       sign(df)
       scale(df)                row-normalise so sum(|values|) == 1
       indneutralize(df, groups)  subtract per-group row mean

   Research-method primitives (causal trailing-window building blocks):
       ts_zscore(df, n)         trailing z-score using the last n bars
       rolling_beta(y, x, n)    trailing OLS beta of y on x
       rolling_residual(y, x, n) current residual from trailing OLS y~x
       rolling_autocorr(df, n, lag) trailing autocorr; use lag=1 unless justified
       signed_area(x, y, n)     two-channel trailing path signed area
       path_length(x, y, n)     two-channel trailing path length
       tail_ratio(df, n, q)     upper-tail magnitude / lower-tail magnitude
       sign_entropy(df, n)      entropy of positive / negative / zero signs

3. DIRECT pandas / numpy / scipy - for plumbing and inline math:
       df.fillna(0.0)   df.where(cond, 0.0)   df.replace(0, np.nan)   df.clip(...)
       df.rolling(n, min_periods=n).std()      df.ewm(halflife=h).mean()
       np.log1p(df)   np.sqrt(df)   scipy.signal.welch(...)   scipy.special.expit(...)
   These are METHODS / functions - call them directly; they are NOT in ``ops``.

IMPORT CONTRACT - the source of almost every retry you keep hitting
-------------------------------------------------------------------
Import ONLY from this set.  Anything else is rejected BEFORE your code runs,
burning a whole attempt:

    from __future__ import annotations
    import pandas as pd
    import numpy as np
    import math                                  # + statistics / functools /
                                                 #   itertools / collections /
                                                 #   dataclasses / typing / warnings
    from scipy import signal, stats, linalg, special, integrate, fft  # pick what you use
    import statsmodels.api as sm                 # causal time-series, robust stats
    from quant_fund_agent.factors.base import BaseFactor
    from quant_fund_agent.factors.registry import register_factor
    from quant_fund_agent.factors.ops import <only names listed above>

Two rules behind the import rejections - internalise both:

  (A) Specialised third-party libraries are NOT installed.  There is NO
      ``iisignature``, ``signatory``, ``tick``, ``hawkeslib``, ``pywt``,
      ``torch``, ``numba``, ``cvxpy``, ``arch``.  Do not import them - implement
      the paper's maths yourself with numpy / scipy / statsmodels.  That is
      exactly why an LLM is doing this and not a grammar-bound search: a
      truncated signature is a few nested cumulative sums; a Hawkes intensity is
      an exponentially-decayed event kernel - write them out.

  (B) ``ops`` exports ONLY the free functions listed above.  Pandas/numpy
      methods are NOT in it.  Never
      ``from quant_fund_agent.factors.ops import fillna`` (or ``where`` /
      ``replace`` / ``dropna`` / ``rolling`` / ``clip`` / ``mean`` / ``std``):
      those are DataFrame methods - call them on the frame, e.g.
      ``df.fillna(0.0)``.

LEAKAGE RULES (they hold for your custom maths too):
    - no negative shifts/diffs/pct_change periods; no centered rolling windows;
    - trailing windows and positive lags only - the value at t sees only <= t;
    - no fitting a *learned* sklearn/statsmodels estimator inside ``calc`` on
      the full panel.  A rolling / trailing OLS you compute yourself IS fine;
      learned models need an explicit fit/predict interface elsewhere.

WRONG - rejected by the validator:
    from quant_fund_agent.factors.ops import fillna    # NO - it's df.fillna(...)
    import iisignature                                 # NO - not installed
RIGHT:
    df = data["close"].fillna(0.0)
    safe = df.replace(0, np.nan)                       # write the maths yourself
"""


# A real seed factor used as a worked example.  Showing one complete,
# working file is more effective than any amount of prose at teaching
# the LLM the expected shape (decorator, class attributes, imports,
# positional ops calls, defensive handling of sparse fields).
EXAMPLE_FACTOR = '''\
EXAMPLE OF A VALID FACTOR FILE
------------------------------
This is a real factor that lives in ``factors/momentum/alpha_007.py``.
Match this file structure exactly: imports, ``@register_factor`` decorator,
class attributes, and a ``calc`` method returning an aligned DataFrame.  The
mechanism does NOT need to look like this example; use richer helper functions
when the paper calls for them.

    """Alpha#7: signed-power of trailing close delta on high-volume bars."""

    from __future__ import annotations

    import pandas as pd
    import numpy as np

    from quant_fund_agent.factors.base import BaseFactor
    from quant_fund_agent.factors.ops import abs_, adv, delta, sign, ts_rank
    from quant_fund_agent.factors.registry import register_factor


    @register_factor
    class Alpha007(BaseFactor):
        factor_id = "alpha_007"
        name = "Alpha#7"
        category = "momentum"
        description = (
            "On high-volume bars (volume > 20-bar ADV) emit a momentum "
            "signal: signed 60-bar rank of |7-bar close delta|.  Otherwise "
            "fall back to a constant -1."
        )
        window_length = 60
        inputs = ["close", "volume"]
        prediction_horizon = 6          # bars ahead the momentum edge peaks
        suggested_horizons = [1, 6, 60]

        def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            close = data["close"]
            volume = data["volume"]
            adv20 = adv(volume, 20)
            d7 = delta(close, 7)
            active = (-1.0 * ts_rank(abs_(d7), 60)) * sign(d7)
            fallback = pd.DataFrame(-1.0, index=close.index, columns=close.columns)
            return active.where(adv20 < volume, fallback)

EXAMPLE OF A CUSTOM PAPER-MATH FACTOR (this is the encouraged pattern)
---------------------------------------------------------------------
Here the mechanism is NOT in ``ops`` — it is a Lo–MacKinlay variance ratio,
written as a small vectorised, causal helper.  This is what "implement the
paper's actual mechanism" looks like: plain numpy/pandas, trailing-only,
deterministic, no special library needed.  Do this whenever a paper's object
of interest (a signature, an intensity, a spectral feature, …) is richer than
the shelf provides.

    """Variance-ratio regime factor: fade names whose returns mean-revert."""

    from __future__ import annotations

    import numpy as np
    import pandas as pd

    from quant_fund_agent.factors.base import BaseFactor
    from quant_fund_agent.factors.ops import rank, returns
    from quant_fund_agent.factors.registry import register_factor


    def _variance_ratio(ret: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
        """Lo–MacKinlay variance ratio over horizon k, estimated on a trailing
        n-bar window.  VR < 1 => mean-reverting, VR > 1 => trending.  Causal:
        every term is a trailing rolling statistic, so value at t sees only <= t."""
        var_1 = ret.rolling(n, min_periods=n).var()
        k_ret = ret.rolling(k, min_periods=k).sum()          # overlapping k-bar returns
        var_k = k_ret.rolling(n, min_periods=n).var()
        return var_k / (k * var_1).replace(0.0, np.nan)


    @register_factor
    class VarianceRatioReversion(BaseFactor):
        factor_id = "variance_ratio_reversion"
        name = "Variance-ratio reversion"
        category = "mean_reversion"
        description = (
            "Cross-sectional rank of (1 - trailing variance ratio): long the "
            "most mean-reverting names, short the most trending ones."
        )
        window_length = 120
        inputs = ["close"]
        prediction_horizon = 12
        suggested_horizons = [6, 12, 24]

        def calc(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
            ret = returns(data)
            vr = _variance_ratio(ret, 5, 100)
            return rank(1.0 - vr)
'''


BRAINSTORM_PROMPT = """\
You are a senior quantitative researcher at a systematic trading firm.
Your job in this session is to invent {n_ideas} new alpha factor(s)
that complement an existing seed ensemble.

DATA CONTEXT
------------
{data_context}

INPUT PAPERS
------------
You have been given the following research papers to draw inspiration
from.  Use them critically — paraphrase or build on the ideas, do not
copy formulas blindly.  You are also free to lean on your own beliefs
about how markets work, as long as the resulting factor is grounded
in the data we actually have.

{papers_block}

PAPER-TO-FACTOR STANDARD
------------------------
Read the paper like a researcher implementing a mechanism, not like a prompt
asking for a familiar formula.  Extract the paper's actual object of interest
(state variable, event process, path shape, residual, tail behaviour, regime,
or interaction), map it to the fields available here, and implement the closest
causal approximation.  You are NOT confined to a fixed operator vocabulary: if
a paper turns on truncated path signatures, Hawkes-process intensities,
spectral / wavelet / entropy features, rolling eigen-structure, or any other
richer transform, propose exactly that — the codegen step implements such maths
directly as vectorised, causal numpy/scipy code, so do not pre-emptively flatten
it into a rank/mean/delta.  You are not limited to minor recombinations of the
existing seed alphas; creative, method-faithful mechanisms are precisely what
this stage is for, as long as they remain computable from the listed fields and
leak-free.

WHAT TO PRODUCE
---------------
Propose exactly {n_ideas} distinct factor ideas.  Each idea must:

  - have a unique snake_case `factor_id` (lowercase, ascii, starting
    with a letter, ≤ 60 chars) that does NOT collide with any of the
    existing factor IDs below;
  - belong to one of the categories: momentum, mean_reversion,
    volatility, microstructure, statistical_arbitrage, carry, sentiment,
    or other;
  - state a clear `trading_idea` — the *why*, in 2-4 sentences
    grounded in market structure, behavioural finance, or the scientific
    method in the paper, citing the source paper(s) if relevant;
  - declare a `prediction_horizon` (positive integer number of bars) —
    the forward offset at which the signal's edge is expected to
    materialise (see the PREDICTION HORIZON note in the DATA CONTEXT);
    optionally a few `suggested_horizons` worth measuring;
  - be cross-sectional (return a DataFrame indexed by time, columns =
    tickers, like the seed alphas);
  - be computable from the fields listed in the DATA CONTEXT above;
  - be reasonably diverse — do not propose two ideas that are minor
    variations of the same signal, and do not merely rename a standard
    rolling mean/rank/correlation as a sophisticated paper concept.

EXISTING FACTOR IDS (do NOT reuse)
----------------------------------
{existing_ids}

OUTPUT FORMAT
-------------
Respond with strict JSON:
{{
  "ideas": [
    {{
      "factor_id": "<snake_case_id>",
      "name": "<short human-readable name>",
      "category": "<one of the categories above>",
      "trading_idea": "<2-4 sentences: why this should have edge>",
      "description": "<1-2 sentences: what the signal computes>",
      "prediction_horizon": <positive int: bars ahead the edge peaks>,
      "suggested_horizons": [<int>, ...],
      "source_paper_ids": ["<paper_id>", ...]
    }},
    ...
  ]
}}
"""


CODEGEN_PROMPT = """\
You are writing production Python for a quant trading firm.

You will produce the full source of ONE factor file, ready to be
saved as ``quant_fund_agent/factors/researcher/{factor_id}.py``.

DATA CONTEXT
------------
{data_context}

{operator_reference}

{example_factor}

FACTOR SPEC
-----------
factor_id:        {factor_id}
name:             {name}
category:         {category}
trading_idea:     {trading_idea}
description:      {description}
prediction_horizon: {prediction_horizon}   (bars ahead the edge peaks)
suggested_horizons: {suggested_horizons}

STRICT REQUIREMENTS
-------------------
1. Import ONLY from these (anything else is rejected BEFORE your code runs — a
   wasted attempt; see the IMPORT CONTRACT above):
     - ``__future__`` (annotations)
     - ``pandas`` (as pd), ``numpy`` (as np)
     - ``math`` and, only if genuinely needed, other pure-stdlib compute
       modules (``statistics``, ``functools``, ``itertools``, ``collections``,
       ``dataclasses``, ``typing``, ``warnings``)
     - ``scipy`` and ``statsmodels`` — your toolbox for the *actual* paper maths
       (signatures, Hawkes intensities, spectral / wavelet features, robust
       stats, causal time-series).  There is NO ``iisignature`` / ``signatory``
       / ``tick`` / ``hawkeslib`` / ``pywt`` / ``torch`` / ``numba`` / ``arch``
       — implement the mechanism yourself from numpy/scipy/statsmodels.
     - ``quant_fund_agent.factors.base`` / ``.registry`` / ``.ops``
   Do NOT fit a *learned* sklearn/statsmodels estimator inside ``calc`` on the
   full panel; a rolling / trailing OLS you compute yourself is fine, and
   learned models need a separate fit/predict interface.
2. Define exactly ONE class subclassing ``BaseFactor``, decorated with
   ``@register_factor``.
3. The class MUST set the following class attributes:
     - ``factor_id = "{factor_id}"``
     - ``category = "{category}"``
     - ``inputs = ["field1", "field2", ...]`` — the EXACT list of data
       fields your ``calc()`` reads via ``data["..."]``.  The agent
       uses this list to decide which fields to load on the panel; if
       a field your code touches is not in ``inputs`` the validator
       will reject the file.
     - ``prediction_horizon = {prediction_horizon}`` — a positive int
       number of bars (the forward offset its edge predicts; see the
       PREDICTION HORIZON note in DATA CONTEXT).  Optionally also
       ``suggested_horizons = [...]`` (a list of positive ints).
4. Implement ``def calc(self, data: dict[str, pd.DataFrame]) ->
   pd.DataFrame`` so the output:
     - has the SAME index and columns as ``data["close"]``;
     - never raises on missing/NaN values — guard sparse fields
       (``effSpread``, ``effLobImb``) with ``.fillna(0.0)``,
       ``.replace(0, np.nan)`` or ``df.where(...)`` as appropriate;
     - returns a numeric DataFrame (signal values, not booleans).
5. Call every helper from ``ops`` **positionally** — they do not accept
   keyword arguments.  Use ``ts_sum(x, 5)``, NOT ``ts_sum(x, n=5)`` or
   ``ts_sum(x, window=5)`` or ``ts_sum(x, 5, min_periods=5)``.
6. For defensive plumbing (NaN handling, masking, replacing zeros,
   dropping rows) call the pandas DataFrame method directly on the
   DataFrame — ``df.fillna(0.0)``, ``df.where(cond, 0.0)``,
   ``df.replace(0, np.nan)``, etc.  These are NOT in ``ops``.  Trying
   to ``from quant_fund_agent.factors.ops import fillna`` will be
   rejected by the validator.
7. Prefer implementing the paper's real mechanism as your OWN vectorised helper
   function(s) in this file over flattening it into a rank/mean/delta.  ``ops``
   is only a convenience shelf; richer maths (path signatures, Hawkes
   intensities, spectral / wavelet features, variance ratios, rolling
   regressions, …) should be written directly with pandas/numpy/scipy/
   statsmodels — see the second worked example above.  Every helper must use
   only information available at time t: trailing windows, positive lags, no
   centered windows, no negative shifts, and no full-panel model fitting.
8. No I/O, no network, no ``os``, no ``open``, no ``eval``/``exec``,
   no ``__import__``.  No filesystem access.
9. Add a short docstring at the top of the file with the trading idea
   verbatim, and a one-line description at the top of the class.

{feedback_block}

OUTPUT FORMAT
-------------
Respond with strict JSON, exactly:
{{
  "code": "<the complete .py file contents as a single string>"
}}

Do NOT wrap the code in markdown fences.  The string must be valid
Python that runs directly.
"""


# Slotted into ``CODEGEN_PROMPT`` on a retry attempt so the LLM sees
# the exact error from the previous try and has a chance to self-correct.
RETRY_FEEDBACK = """\
PREVIOUS ATTEMPT FAILED
-----------------------
Your previous response was rejected with this exact error:

    {error}

Most common causes, in order of frequency:

1. Forgot to declare ``inputs = [...]`` at the class level (or the
   list doesn't cover every ``data["X"]`` referenced in calc()).
   ``inputs`` MUST enumerate every data field your code reads.

   Or forgot ``prediction_horizon = <positive int>`` at the class level
   (a bar count; 1 ≤ it ≤ a few hundred for most signals).

2. Tried to import a pandas DataFrame method from ``ops``
   (``from quant_fund_agent.factors.ops import fillna`` / ``where`` /
   ``replace`` / ``dropna``).  These are NOT in ``ops`` — they are
   methods on the DataFrame.  Use ``df.fillna(0.0)``, ``df.where(...)``,
   ``df.replace(0, np.nan)`` etc. directly on the DataFrame.

3. Imported a library that is NOT installed (e.g. ``iisignature``,
   ``signatory``, ``tick``, ``hawkeslib``, ``pywt``, ``torch``, ``numba``,
   ``arch``).  Only pandas, numpy, math (+ stdlib compute modules), scipy,
   statsmodels and the ``quant_fund_agent.factors.*`` modules are importable —
   implement the paper's maths yourself with numpy/scipy/statsmodels.

4. Passed a keyword argument to an ``ops`` function.  Every op is
   positional-only — ``ts_sum(x, 5)``, NOT ``ts_sum(x, n=5)`` or
   ``ts_sum(x, window=5, min_periods=5)``.

5. Referenced a ``data[...]`` field that does not exist.  Only the
   fields listed in DATA CONTEXT are available.

6. Used a future-looking construct (negative ``shift`` / ``diff`` /
   ``pct_change`` period, centered rolling window, or a ``fit()`` call inside
   ``calc``).  Rewrite it as a trailing-window deterministic transform.

Re-read OPERATOR REFERENCE and DATA CONTEXT carefully, fix the
specific issue named in the error message above, and produce the
complete corrected file.
"""
