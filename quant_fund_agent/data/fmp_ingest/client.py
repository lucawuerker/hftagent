"""Rate-limited, concurrent HTTP client for the one-time FMP archive pull.

``providers/_http.py`` stays as it is — it serves the *live* path, where a single
serial request with a fixed minimum interval is exactly right.  A 40 000-call
bulk download needs different machinery: a real calls-per-minute budget, worker
threads, and — most importantly — the ability to tell a **plan restriction**
(HTTP 402/403: retrying is pointless and the endpoint should be marked
unavailable) apart from a **rate limit** (HTTP 429: back off and retry) and a
transient network error.

Every response is classified into :class:`FetchResult`; nothing here raises for a
per-request failure, because one bad symbol must never abort a multi-hour run.
Byte counts are accumulated so the run can be checked against FMP's trailing
30-day bandwidth cap (50 GB on Premium).
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import requests

log = logging.getLogger("data.fmp_ingest.client")

_HEADERS = {"User-Agent": "QuantFundAgent/1.0 (academic research)"}

#: Substrings FMP uses in its plan-restriction bodies (it answers 402, and
#: occasionally 403, with a plain-text explanation rather than JSON).
_RESTRICTION_MARKERS = (
    "Restricted Endpoint",
    "Premium Query Parameter",
    "not available under your current subscription",
    "Exclusive Endpoint",
    "Special Endpoint",
)


class RateLimiter:
    """Exact sliding-window limiter: at most ``per_minute`` calls in any 60 s.

    A deque of send timestamps is cheaper and more accurate than a token bucket
    at this scale, and it never lets a burst of workers exceed the plan cap (the
    failure mode that gets a key throttled for the rest of the run).
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = max(1, int(per_minute))
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0])
            time.sleep(max(wait, 0.01))


@dataclass
class FetchResult:
    """Outcome of one request — never an exception, always a verdict."""

    status: int = 0                 # HTTP status; 0 = network/parse failure
    rows: list[dict] = field(default_factory=list)
    n_bytes: int = 0
    error: str | None = None
    restricted: bool = False        # plan-gated: do not retry, mark unavailable

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def empty(self) -> bool:
        return self.ok and not self.rows


class FMPClient:
    """Thread-safe FMP client with a shared calls-per-minute budget.

    ``api_key`` is injected per request and never logged; only paths and params
    minus the key appear in log lines.
    """

    def __init__(
        self,
        api_key: str,
        *,
        per_minute: int = 600,
        timeout: float = 60.0,
        retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("FMP_API_KEY not set in .env (see .env.example).")
        self._key = api_key
        self.limiter = RateLimiter(per_minute)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self.calls = 0
        self.bytes = 0
        self.restricted_calls = 0
        self.failed_calls = 0

    # ── internals ───────────────────────────────────────────────────────────

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update(_HEADERS)
            self._local.session = s
        return s

    def _record(self, result: FetchResult) -> None:
        with self._stats_lock:
            self.calls += 1
            self.bytes += result.n_bytes
            if result.restricted:
                self.restricted_calls += 1
            elif not result.ok:
                self.failed_calls += 1

    @staticmethod
    def _classify_body(text: str) -> str | None:
        """Return a restriction message if the body is a plan-gate notice."""
        head = text[:400]
        for marker in _RESTRICTION_MARKERS:
            if marker in head:
                return head.strip()
        return None

    # ── public API ──────────────────────────────────────────────────────────

    def get(self, url: str, params: dict) -> FetchResult:
        """GET ``url``; retries transient failures, never raises for the caller."""
        query = {**params, "apikey": self._key}
        last_error: str | None = None
        status = 0
        for attempt in range(1, self.retries + 1):
            self.limiter.acquire()
            try:
                resp = self._session().get(url, params=query, timeout=self.timeout)
            except Exception as e:  # noqa: BLE001 — network hiccup: retry
                last_error = f"{type(e).__name__}: {e}"
                status = 0
            else:
                status = resp.status_code
                body = resp.text or ""
                n_bytes = len(resp.content or b"")

                if status in (401, 402, 403):
                    msg = self._classify_body(body) or f"HTTP {status}"
                    out = FetchResult(status=status, n_bytes=n_bytes,
                                      error=msg, restricted=True)
                    self._record(out)
                    return out

                if status == 429:
                    last_error = "HTTP 429 (rate limited)"
                    # Respect a server-provided window when there is one.
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = float(retry_after) if (retry_after or "").isdigit() else 15.0 * attempt
                    log.warning("fmp: 429 — backing off %.0fs (attempt %d/%d)",
                                sleep_s, attempt, self.retries)
                    time.sleep(sleep_s)
                    continue

                if status != 200:
                    last_error = f"HTTP {status}: {body[:200]}"
                else:
                    try:
                        payload = resp.json()
                    except (json.JSONDecodeError, ValueError) as e:
                        last_error = f"unparseable JSON: {e}"
                    else:
                        # FMP signals some errors with a dict instead of an array.
                        if isinstance(payload, dict):
                            err = payload.get("Error Message") or payload.get("error")
                            if err:
                                out = FetchResult(status=status, n_bytes=n_bytes,
                                                  error=str(err))
                                self._record(out)
                                return out
                            rows = [payload]
                        elif isinstance(payload, list):
                            rows = [r for r in payload if isinstance(r, dict)]
                        else:
                            rows = []
                        out = FetchResult(status=200, rows=rows, n_bytes=n_bytes)
                        self._record(out)
                        return out

            if attempt < self.retries:
                backoff = min(30.0, 1.5 * attempt) + random.uniform(0, 0.5)
                log.debug("fmp: %s — retrying in %.1fs (attempt %d/%d)",
                          last_error, backoff, attempt, self.retries)
                time.sleep(backoff)

        out = FetchResult(status=status, error=last_error or "unknown failure")
        self._record(out)
        return out

    def stats(self) -> dict[str, float]:
        with self._stats_lock:
            return {
                "calls": self.calls,
                "bytes": self.bytes,
                "mb": round(self.bytes / 1e6, 1),
                "restricted": self.restricted_calls,
                "failed": self.failed_calls,
            }
