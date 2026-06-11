"""Optional LLM layer for the onboarding wizard (``setup.py --assist``).

Turns a plain-English description ("mid-cap US tech, last 2 years, daily") into a
**proposed** config that the deterministic wizard then shows for confirmation.

Design contract (deliberately conservative):

* The proposal is only ever a set of *suggested defaults*. The wizard's
  precedence stays **explicit CLI flag > LLM proposal > built-in default**, and
  in interactive mode every field is still shown for the user to accept/override.
* Every value the model returns is **validated against ground truth** (usable
  providers, existing presets, parseable dates). Anything unknown is dropped, so
  the LLM can only ever *narrow* to legal values — never inject a bad one.
* Any failure (no ``OPENAI_API_KEY``, network down, unparseable output) returns
  ``{}`` so the wizard falls straight back to its deterministic path. The core
  wizard never depends on an LLM key.

The LLM is built through :func:`quant_fund_agent.llm.make_chat_llm`, so the same
provider-agnostic factory (and ``FACTOR_RESEARCH_LLM_*`` env) the agents use also
drives onboarding. Override the model for assist alone with
``SETUP_ASSIST_LLM_MODEL``.
"""

from __future__ import annotations

import json
import os
from datetime import date

# Keys the wizard understands; the model is told to emit a subset of these.
_PROPOSAL_KEYS = (
    "provider", "asset_class", "frequency",
    "start", "end", "preset", "tickers", "n_tickers",
)


def _system_prompt(*, available: list[str], presets: list[str], today: str) -> str:
    return (
        "You configure a quantitative backtest by translating a user's plain-English "
        "description into a STRICT JSON object. Emit ONLY the JSON, no prose, no code "
        "fences.\n\n"
        f"Today's date is {today}. Resolve relative ranges (e.g. 'last 2 years', "
        "'since 2020') into absolute ISO dates (YYYY-MM-DD) computed from today.\n\n"
        "Allowed keys (ALL optional — omit any you are unsure about):\n"
        f"  provider     one of {available} (pick the one the user names, else omit)\n"
        "  asset_class  'equity' (only equities are supported today)\n"
        "  frequency    '1d' (only daily is supported today)\n"
        "  start        ISO date YYYY-MM-DD\n"
        "  end          ISO date YYYY-MM-DD\n"
        f"  preset       one of {presets} (a named universe)\n"
        "  tickers      a JSON list of US equity symbols, e.g. [\"AAPL\",\"MSFT\"] "
        "(use this OR preset, not both; prefer it only when the user names symbols)\n"
        "  n_tickers    positive integer cap on universe size\n\n"
        "Rules: never invent a provider or preset outside the allowed lists; if the "
        "user describes a sector/theme and you do not have a matching preset, return a "
        "small explicit 'tickers' list instead. Keep 'tickers' to at most 15 symbols."
    )


def _extract_json(text: str) -> str | None:
    """Pull the first ``{...}`` block out of an LLM reply (tolerant of fences/prose)."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _coerce_content(resp) -> str:
    """langchain messages carry ``.content`` (str, or a list of parts for some models)."""
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def validate_proposal(
    raw: dict, *, available: list[str], presets: list[str]
) -> dict:
    """Keep only well-formed, in-bounds keys from a raw model proposal."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}

    prov = raw.get("provider")
    if isinstance(prov, str) and prov in available:
        out["provider"] = prov

    ac = raw.get("asset_class")
    if isinstance(ac, str) and ac:
        out["asset_class"] = ac

    freq = raw.get("frequency")
    if isinstance(freq, str) and freq:
        out["frequency"] = freq

    start = _valid_date(raw.get("start"))
    end = _valid_date(raw.get("end"))
    if start and end and start < end:
        out["start"], out["end"] = start, end  # only accept a coherent pair

    preset = raw.get("preset")
    if isinstance(preset, str) and preset in presets:
        out["preset"] = preset

    tickers = raw.get("tickers")
    if isinstance(tickers, list):
        clean = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if clean:
            out["tickers"] = clean[:15]
            out.pop("preset", None)  # tickers and preset are mutually exclusive

    n = raw.get("n_tickers")
    if isinstance(n, int) and n > 0:
        out["n_tickers"] = n
    elif isinstance(n, str) and n.isdigit() and int(n) > 0:
        out["n_tickers"] = int(n)

    return out


def _valid_date(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError:
        return None


def propose_config(
    description: str,
    *,
    available: list[str],
    presets: list[str],
    model: str | None = None,
) -> dict:
    """Best-effort LLM proposal for ``description``; ``{}`` on any failure.

    Returns a partial mapping over :data:`_PROPOSAL_KEYS`, already validated
    against ``available`` providers and existing ``presets``.
    """
    description = (description or "").strip()
    if not description:
        return {}

    try:
        from quant_fund_agent.llm import make_chat_llm

        model = model or os.getenv("SETUP_ASSIST_LLM_MODEL")
        llm = make_chat_llm(model=model, temperature=0)
        messages = [
            ("system", _system_prompt(
                available=available, presets=presets, today=date.today().isoformat())),
            ("human", description),
        ]
        resp = llm.invoke(messages)
        blob = _extract_json(_coerce_content(resp))
        if not blob:
            return {}
        raw = json.loads(blob)
    except Exception:  # noqa: BLE001 — assist is strictly optional; never crash onboarding
        return {}

    return validate_proposal(raw, available=available, presets=presets)
