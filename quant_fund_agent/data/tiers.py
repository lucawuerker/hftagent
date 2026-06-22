"""Data capability tiers — named groups of panel fields.

A *tier* is a named set of fields a data provider may supply.  Each
:class:`~quant_fund_agent.data.providers.base.DataProvider` declares the union of
tiers it can fill via ``available_fields()``; each factor declares the fields it
needs via ``BaseFactor.inputs``.  A factor is **compatible** with a provider iff
its inputs are a subset of the provider's available fields — otherwise it is
gated out (Phase 2) rather than crashing on a missing column.

The tiers are modelled as **named sets**, not a strict linear order: a vendor can
supply microstructure without fundamentals, or vice versa.  ``standard`` is the
common base every equity vendor can fill.

``vwap`` and ``returns`` live in ``standard`` because the panel layer
**synthesizes** them from OHLCV when a provider doesn't supply them directly
(``vwap ≈ (high + low + close) / 3``, ``returns = close.pct_change()``), so any
OHLCV provider satisfies factors that need them.
"""

from __future__ import annotations

from quant_fund_agent.data.fields import (
    ESTIMATE_FIELDS,
    EVENT_FIELDS,
    FUNDAMENTAL_FIELDS,
)

# Per-level order-book fields the LOBSTER converter can emit
# (``askPrice{i}``/``askDepth{i}``/``bidPrice{i}``/``bidDepth{i}``).  LOBSTER tops
# out at 10 levels, so the microstructure tier advertises that bound; a shallower
# pull simply leaves the deeper fields NaN (factors degrade, never crash — same
# as the fundamentals tiers on a free vendor plan).
_MAX_BOOK_LEVELS = 10
_LEVEL_FIELDS: frozenset[str] = frozenset(
    f"{side}{i}"
    for i in range(1, _MAX_BOOK_LEVELS + 1)
    for side in ("askPrice", "askDepth", "bidPrice", "bidDepth")
)

# Each tier lists the fields it ADDS; ``available_fields`` for a provider is the
# union of the tiers it can fill.  ``fundamental`` / ``estimates`` / ``events``
# carry the non-OHLCV vocabulary from :mod:`quant_fund_agent.data.fields` (kept
# there so the normalization maps and the tier sets can't drift apart).
TIERS: dict[str, frozenset[str]] = {
    "standard": frozenset(
        {"open", "high", "low", "close", "volume", "vwap", "returns"}
    ),
    "fundamental": FUNDAMENTAL_FIELDS,
    "estimates": ESTIMATE_FIELDS,
    "events": EVENT_FIELDS,
    "microstructure": frozenset(
        {
            "trade", "orderFlow", "hidden", "auction",
            "spread", "effSpread", "lobImb", "effLobImb",
            "trdLiq", "ofLiq", "depth",
            "nbEvents", "nbHidden", "nbTrades",
        }
    )
    | _LEVEL_FIELDS,
}

# Fields the panel layer can synthesize from plain OHLCV (so they never gate a
# factor out on a standard provider).
SYNTHESIZED_FIELDS: frozenset[str] = frozenset({"vwap", "returns"})

# ── LOBSTER data level (order-book "Level 2" vs message-stream "Level 3") ────
#
# A LOBSTER export can be a plain limit-order book (Level 2) or the full
# order/message stream (Level 3).  These two sets split the microstructure tier
# by which level is needed:
#
#   • Level 2 — derivable from the order book alone: per-level prices/sizes, the
#     spread, total depth and book imbalances.  (Traded ``volume`` and the
#     mid-derived OHLC live in the ``standard`` tier and are treated as available
#     alongside the book, so a Level-2 feed keeps them.)
#   • Level 3 — needs the trade/message stream and so CANNOT be reconstructed
#     from the book: signed trades, order flow, hidden & auction prints, the
#     effective spread, trade/event counts and the trade-side liquidity proxies.
LOBSTER_L2_MICRO_FIELDS: frozenset[str] = (
    frozenset({"spread", "depth", "lobImb", "effLobImb"}) | _LEVEL_FIELDS
)
LOBSTER_L3_MICRO_FIELDS: frozenset[str] = TIERS["microstructure"] - LOBSTER_L2_MICRO_FIELDS


def lobster_fields_for_level(level: int) -> frozenset[str]:
    """Fields a LOBSTER feed exposes at order-book ``level`` (2 or 3).

    Level 2 = the ``standard`` tier (mid-derived OHLC + traded volume) plus every
    book-derivable microstructure field.  Level 3 additionally exposes the
    message-stream fields (:data:`LOBSTER_L3_MICRO_FIELDS`).  Anything ``>= 3``
    is treated as full Level 3; anything else as Level 2.
    """
    fields = TIERS["standard"] | LOBSTER_L2_MICRO_FIELDS
    if int(level) >= 3:
        fields = fields | LOBSTER_L3_MICRO_FIELDS
    return frozenset(fields)

# Tier "rank" only for choosing a human-readable label for a factor's required
# tier (display/filtering).  Higher = richer data requirement.
_TIER_RANK = {
    "standard": 0,
    "fundamental": 1,
    "estimates": 2,
    "events": 3,
    "microstructure": 4,
}


def all_known_fields() -> frozenset[str]:
    """Every field across every tier."""
    out: set[str] = set()
    for fields in TIERS.values():
        out |= fields
    return frozenset(out)


def field_tier(field: str) -> str | None:
    """Which tier a field belongs to (``None`` if unknown)."""
    for tier, fields in TIERS.items():
        if field in fields:
            return tier
    return None


def tiers_for_fields(fields: frozenset[str]) -> frozenset[str]:
    """Union of the named tiers a provider must support to supply ``fields``."""
    needed: set[str] = set()
    for f in fields:
        tier = field_tier(f)
        if tier is not None:
            needed.add(tier)
    return frozenset(needed)


def required_tier(inputs: list[str] | None) -> str:
    """Human-readable label for the richest tier a factor's ``inputs`` touch.

    Unknown fields are ignored for labelling; an empty/closed-OHLCV factor is
    ``"standard"``.
    """
    rank = 0
    label = "standard"
    for f in inputs or []:
        tier = field_tier(f)
        if tier is not None and _TIER_RANK[tier] > rank:
            rank = _TIER_RANK[tier]
            label = tier
    return label


def is_compatible(inputs: list[str] | None, available: frozenset[str] | set[str]) -> bool:
    """True iff every field a factor needs is supplied by the provider.

    ``SYNTHESIZED_FIELDS`` are always considered satisfiable because the panel
    layer derives them from OHLCV.
    """
    needed = set(inputs or [])
    return needed <= (set(available) | set(SYNTHESIZED_FIELDS))


def resolve_required_inputs(record: object) -> list[str]:
    """Best-effort list of fields a factor record needs, for gating.

    Resolution order (conservative — admit rather than over-gate):
      1. the record's persisted ``required_inputs`` (set at creation / backfill);
      2. the live factor class's ``inputs`` from the registry;
      3. ``["close"]`` (treat as plain ``standard``).

    ``record`` may be a ``FactorRecord``-like object or a plain dict (the catalog
    reads raw JSON), so attributes are looked up defensively.
    """
    def _get(key: str):
        if isinstance(record, dict):
            return record.get(key)
        return getattr(record, key, None)

    stored = _get("required_inputs")
    if stored:
        return list(stored)

    fid = _get("id") or _get("factor_id")
    if fid:
        try:
            from quant_fund_agent.factors.registry import get_factor_class

            cls = get_factor_class(fid)
            if cls is not None:
                inputs = getattr(cls, "inputs", None)
                if inputs:
                    return list(inputs)
        except Exception:
            pass
    return ["close"]


def compatible_factors(
    records: list,
    available: frozenset[str] | set[str],
) -> list:
    """Filter records to those whose required fields the provider supplies."""
    return [r for r in records if is_compatible(resolve_required_inputs(r), available)]
