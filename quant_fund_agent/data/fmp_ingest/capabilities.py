"""What the *actual* subscription serves — probed once, then obeyed.

FMP gates by plan at three different levels, and all three matter for a bulk
pull:

1. **whole endpoints** (``historical-sp500-constituent``, ``symbol-change``);
2. **parameter values** — ``period=quarter``, ``from``/``to`` on
   ``historical-market-capitalization``, and a numeric cap on ``limit``
   (*"the values for 'limit' must be between 0 and 5"*);
3. **individual symbols** — delisted tickers (``AABA``, ``ENRNQ``) answer 402
   *"this value set for 'symbol' is not available under your current
   subscription"* on lower plans even though the data exists.

Rather than hard-coding a plan matrix that goes stale, the probe hits each
registered endpoint once with the live key and writes ``capabilities.json``.  The
downloader then only spends calls on what the plan actually serves, and the
report tells you exactly what your key is missing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from quant_fund_agent.data.fmp_ingest.client import FMPClient, FetchResult
from quant_fund_agent.data.fmp_ingest.endpoints import ENDPOINTS, Endpoint

log = logging.getLogger("data.fmp_ingest.capabilities")

CAPABILITIES_FILE = "capabilities.json"

#: A liquid, long-listed name every plan serves — the probe's control symbol.
PROBE_SYMBOL = "AAPL"
#: Delisted S&P 500 members: the acid test for closing the survivorship gap.
PROBE_DELISTED = ("AABA", "ENRNQ", "VIAC")

_LIMIT_CAP_RE = re.compile(r"must be between\s+\d+\s+and\s+(\d+)", re.I)


@dataclass
class Capabilities:
    """Probe outcome; also the runtime policy object the downloader consults."""

    endpoints: dict[str, dict] = field(default_factory=dict)
    limit_cap: int | None = None            # plan cap on `limit`, None = uncapped
    quarterly_ok: bool = True
    delisted_ok: bool = False
    delisted_probe: dict[str, str] = field(default_factory=dict)
    price_full_range_ok: bool = True        # one request covers the whole window
    price_earliest: str | None = None
    constituent_earliest: str | None = None
    probed_at: str = ""

    def allows(self, endpoint: str | Endpoint) -> bool:
        name = endpoint if isinstance(endpoint, str) else endpoint.name
        info = self.endpoints.get(name)
        return True if info is None else info.get("status") != "restricted"

    def limit_for(self, endpoint: Endpoint) -> int | None:
        if endpoint.limit is None:
            return None
        if self.limit_cap is None:
            return endpoint.limit
        return min(endpoint.limit, self.limit_cap)

    def restricted(self) -> list[str]:
        return sorted(n for n, i in self.endpoints.items() if i.get("status") == "restricted")

    def to_dict(self) -> dict:
        return {
            "probed_at": self.probed_at,
            "limit_cap": self.limit_cap,
            "quarterly_ok": self.quarterly_ok,
            "delisted_ok": self.delisted_ok,
            "delisted_probe": self.delisted_probe,
            "price_full_range_ok": self.price_full_range_ok,
            "price_earliest": self.price_earliest,
            "constituent_earliest": self.constituent_earliest,
            "endpoints": self.endpoints,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Capabilities":
        return cls(
            endpoints=payload.get("endpoints") or {},
            limit_cap=payload.get("limit_cap"),
            quarterly_ok=bool(payload.get("quarterly_ok", True)),
            delisted_ok=bool(payload.get("delisted_ok", False)),
            delisted_probe=payload.get("delisted_probe") or {},
            price_full_range_ok=bool(payload.get("price_full_range_ok", True)),
            price_earliest=payload.get("price_earliest"),
            constituent_earliest=payload.get("constituent_earliest"),
            probed_at=payload.get("probed_at", ""),
        )


def _probe_params(ep: Endpoint, start: str, end: str) -> dict:
    """A minimal, representative request for one endpoint."""
    params: dict[str, str | int] = dict(ep.params)
    if ep.kind == "symbol":
        params["symbol"] = PROBE_SYMBOL
    if ep.windowed:
        params["from"], params["to"] = start, end
    if ep.limit is not None:
        params["limit"] = ep.limit
    if ep.periods:
        params["period"] = ep.periods[0]
    if ep.paginated:
        params["page"] = 0
    return params


def _summarise(result: FetchResult) -> dict:
    if result.restricted:
        status = "restricted"
    elif result.ok:
        status = "ok"
    else:
        status = "error"
    info = {"status": status, "http": result.status, "rows": len(result.rows)}
    if result.error:
        info["message"] = result.error[:240]
    return info


def _dates(rows: list[dict], key: str = "date") -> list[pd.Timestamp]:
    stamps = [pd.to_datetime(r.get(key), errors="coerce") for r in rows if key in r]
    return sorted(s for s in stamps if pd.notna(s))


def probe(
    client: FMPClient,
    *,
    start: str = "2004-01-01",
    end: str | None = None,
    names: list[str] | None = None,
) -> Capabilities:
    """Call every registered endpoint once and derive the plan's real limits."""
    end = end or datetime.now(timezone.utc).date().isoformat()
    caps = Capabilities(probed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    wanted = names or list(ENDPOINTS)

    for name in wanted:
        ep = ENDPOINTS[name]
        result = client.get(ep.url(), _probe_params(ep, start, end))
        caps.endpoints[name] = _summarise(result)

        # A `limit` cap is reported in the restriction text; keep the tightest.
        if result.restricted and result.error:
            match = _LIMIT_CAP_RE.search(result.error)
            if match:
                cap = int(match.group(1))
                caps.limit_cap = cap if caps.limit_cap is None else min(caps.limit_cap, cap)
                # Re-probe within the cap so the endpoint isn't wrongly written off.
                retry = dict(_probe_params(ep, start, end))
                retry["limit"] = cap
                result = client.get(ep.url(), retry)
                caps.endpoints[name] = _summarise(result)
                caps.endpoints[name]["limit_cap"] = cap

        if name == "prices_adjusted" and result.ok:
            stamps = _dates(result.rows)
            if stamps:
                caps.price_earliest = str(stamps[0].date())
                # One request must reach back to `start`, else chunk the window.
                caps.price_full_range_ok = stamps[0] <= pd.Timestamp(start) + pd.Timedelta(days=30)
        if name == "historical_sp500_constituent" and result.ok:
            stamps = _dates(result.rows, "dateAdded") or _dates(result.rows)
            if stamps:
                caps.constituent_earliest = str(stamps[0].date())

    # Quarterly periods are gated on lower plans; check on a statement endpoint.
    stmt = ENDPOINTS["income_statement"]
    q_params = {"symbol": PROBE_SYMBOL, "period": "quarter",
                "limit": caps.limit_for(stmt) or 5}
    q_result = client.get(stmt.url(), q_params)
    caps.quarterly_ok = q_result.ok and bool(q_result.rows)
    if not caps.quarterly_ok and q_result.error:
        log.warning("fmp: quarterly periods unavailable — %s", q_result.error[:160])

    # Delisted symbols: the whole point of a survivorship-bias-free pull.
    px = ENDPOINTS["prices_adjusted"]
    for sym in PROBE_DELISTED:
        res = client.get(px.url(), {"symbol": sym, "from": start, "to": end})
        caps.delisted_probe[sym] = (
            "restricted" if res.restricted else ("ok" if res.rows else "empty")
        )
    caps.delisted_ok = any(v == "ok" for v in caps.delisted_probe.values())
    return caps


def format_report(caps: Capabilities) -> str:
    """Human-readable probe summary (printed by the CLI, kept in the docs)."""
    lines = [
        "FMP capability probe",
        f"  probed_at            : {caps.probed_at}",
        f"  limit cap            : {caps.limit_cap if caps.limit_cap else 'uncapped'}",
        f"  quarterly periods    : {'yes' if caps.quarterly_ok else 'NO (plan-gated)'}",
        f"  delisted symbols     : {'yes' if caps.delisted_ok else 'NO (survivorship gap stays open)'}"
        f"  {caps.delisted_probe}",
        f"  earliest price bar   : {caps.price_earliest}",
        f"  one-shot price range : {'yes' if caps.price_full_range_ok else 'no — chunking windows'}",
        f"  constituent log from : {caps.constituent_earliest}",
        "",
        "  endpoint                          status     rows",
    ]
    for name in sorted(caps.endpoints):
        info = caps.endpoints[name]
        lines.append(f"    {name:<32}{info.get('status',''):<11}{info.get('rows',0)}")
    restricted = caps.restricted()
    if restricted:
        lines += ["", f"  RESTRICTED ({len(restricted)}): {', '.join(restricted)}"]
    return "\n".join(lines)
