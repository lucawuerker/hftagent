"""Label-frame resolution for ``IndNeutralize`` alphas.

The canonical categorical fields are ``sector`` / ``industry`` /
``subindustry`` (see ``quant_fund_agent.data.fields.CATEGORICAL_FIELDS``);
the ``fmp_archive`` provider serves them as wide object-dtype frames
(dates × tickers of labels), which ``ops.indneutralize`` accepts directly.

``neutralize`` applies ``indneutralize`` using the *finest available* label
field at or above the requested granularity: an alpha written against
``subindustry`` falls back to ``industry`` then ``sector`` when the finer
label is entirely missing (logged once per alpha), and if no label field is
available at all the neutralisation step is skipped with a warning — the
historical Alpha#48/#58/#59 fallback behaviour.
"""

from __future__ import annotations

import logging

import pandas as pd

from quant_fund_agent.factors.ops import indneutralize

_log = logging.getLogger(__name__)

_FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    "subindustry": ("subindustry", "industry", "sector"),
    "industry": ("industry", "sector"),
    "sector": ("sector",),
}

# (alpha_name, message-key) pairs already warned about — log once, not per bar.
_warned: set[tuple[str, str]] = set()


def _has_labels(frame: object) -> bool:
    """True when the field is present and carries at least one label."""
    if isinstance(frame, pd.DataFrame):
        return bool(frame.notna().any().any())
    if isinstance(frame, pd.Series):
        return bool(frame.notna().any())
    return False


def _warn_once(alpha_name: str, key: str, message: str) -> None:
    if (alpha_name, key) not in _warned:
        _warned.add((alpha_name, key))
        _log.warning(message)


def resolve_labels(
    data: dict[str, pd.DataFrame],
    preferred: str,
    alpha_name: str,
) -> tuple[pd.DataFrame | pd.Series | None, str | None]:
    """Return ``(label_frame, field_name)`` for the finest available label
    field, walking ``preferred`` → coarser; ``(None, None)`` if none exist."""
    for field in _FALLBACK_CHAIN[preferred]:
        frame = data.get(field)
        if _has_labels(frame):
            if field != preferred:
                _warn_once(
                    alpha_name,
                    f"{preferred}->{field}",
                    f"{alpha_name}: data[{preferred!r}] missing/empty — "
                    f"falling back to {field!r} for IndNeutralize.",
                )
            return frame, field
    _warn_once(
        alpha_name,
        f"{preferred}->none",
        f"{alpha_name}: no {preferred!r} (or coarser) label field available — "
        "skipping industry neutralisation.",
    )
    return None, None


def neutralize(
    df: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    preferred: str,
    alpha_name: str,
) -> pd.DataFrame:
    """``IndNeutralize(df, <finest available grouping>)``; ``df`` unchanged
    (with a one-time warning) when no label field is available."""
    labels, _field = resolve_labels(data, preferred, alpha_name)
    if labels is None:
        return df
    return indneutralize(df, labels)
