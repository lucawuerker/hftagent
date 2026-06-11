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
    ),
}

# Fields the panel layer can synthesize from plain OHLCV (so they never gate a
# factor out on a standard provider).
SYNTHESIZED_FIELDS: frozenset[str] = frozenset({"vwap", "returns"})

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
