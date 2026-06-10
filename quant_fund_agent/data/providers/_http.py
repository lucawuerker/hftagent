"""Shared HTTP helper for keyed REST data providers (FMP / AlphaVantage).

Centralises the bits every vendor pull needs: a per-provider minimum interval
between requests (to respect free-tier rate limits), retry/backoff on transient
errors, a finite timeout, and a single place to keep the API key out of logs.
"""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger("data.providers.http")

_HEADERS = {"User-Agent": "QuantFundAgent/1.0 (academic research)"}

# Per-provider timestamp of the last request, so we can space calls out.
_LAST_REQUEST: dict[str, float] = {}


class RateLimited(RuntimeError):
    """Raised when a provider signals we've exhausted its quota."""


def _throttle(provider: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    last = _LAST_REQUEST.get(provider, 0.0)
    wait = min_interval - (time.time() - last)
    if wait > 0:
        log.debug("%s: throttling %.1fs to respect rate limit", provider, wait)
        time.sleep(wait)


def request_json(
    url: str,
    params: dict,
    *,
    provider: str,
    min_interval: float = 0.0,
    retries: int = 3,
    timeout: float = 30.0,
) -> dict | list:
    """GET ``url`` and return parsed JSON, with throttling + retry/backoff.

    Raises :class:`RateLimited` on an HTTP 429, and ``RuntimeError`` on a
    persistent non-200 or unparseable body.  Vendor-specific "rate limited"
    JSON payloads (AlphaVantage) are detected by the caller, which can re-raise
    :class:`RateLimited` for a clean message.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        _throttle(provider, min_interval)
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
            _LAST_REQUEST[provider] = time.time()
            if resp.status_code == 429:
                raise RateLimited(f"{provider}: HTTP 429 (rate limited)")
            resp.raise_for_status()
            return resp.json()
        except RateLimited:
            raise
        except Exception as e:  # noqa: BLE001 — retry transient network/JSON errors
            last_err = e
            if attempt < retries:
                backoff = 1.5 * attempt
                log.warning("%s: request failed (attempt %d/%d): %s — retrying in %.1fs",
                            provider, attempt, retries, e, backoff)
                time.sleep(backoff)
    raise RuntimeError(f"{provider}: request to {url} failed after {retries} attempts: {last_err}")
