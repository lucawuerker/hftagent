"""Provider-agnostic chat-LLM factory.

The agents historically each constructed ``langchain_openai.ChatOpenAI`` directly,
hard-wiring OpenAI.  To compare *different* research models — including non-OpenAI
ones — in the factor-creation phase, the factor researcher builds its LLM through
:func:`make_chat_llm`, which wraps langchain's multi-provider
``init_chat_model`` and infers the provider from the model id when not given.

Only the factor researcher is wired to this today; the factory is intentionally
generic so the other agents can adopt it later.

Non-OpenAI providers need their langchain integration package installed
(``langchain-anthropic``, ``langchain-google-genai``, ``langchain-aws``, …);
:func:`make_chat_llm` raises a clear, actionable error if it is missing.

Amazon Bedrock
--------------
Bedrock model ids (``anthropic.claude-…``, ``us.anthropic.…``, ``eu.anthropic.…``,
``meta.llama…``, ``amazon.…``) are inferred to the ``bedrock_converse`` provider
(``langchain-aws``).  Credentials come from the standard AWS chain
(``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``/``AWS_DEFAULT_REGION``,
profile, or instance role) — no ``OPENAI_API_KEY`` needed.

OpenAI-compatible endpoints
---------------------------
Any OpenAI-compatible endpoint (e.g. the Meta Model API for Muse Spark) can be
reached with ``provider="openai"`` plus a base URL: set the env
``FACTOR_RESEARCH_LLM_BASE_URL`` (or pass ``base_url=`` to
:func:`make_chat_llm`, which wins over the env) and point ``OPENAI_API_KEY`` at
that vendor's key.

Token / cost metering
---------------------
Every model built here carries a callback that meters usage into a module-level,
thread-safe :class:`UsageMeter` — per role and in total: calls, input/output
tokens, USD cost (``DEFAULT_PRICES`` merged with the ``QF_LLM_PRICES`` env JSON)
and errors.  Attribute calls to a role with the :func:`set_llm_role` context
manager (or bind a default via ``make_chat_llm(role=...)``); read/reset with
:func:`usage_summary` / :func:`reset_usage`.  Setting ``QF_MAX_LLM_COST_USD``
turns the meter into a hard budget: crossing it raises
:class:`LLMBudgetExceeded`.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

# Model-id prefix → langchain ``model_provider`` token.  Checked in order.
_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
    # Bedrock model ids (vendor-prefixed, dot-separated) — before the bare
    # "claude" prefix so "anthropic.claude-…" routes to Bedrock, "claude-…"
    # to the Anthropic API.
    ("us.anthropic.", "bedrock_converse"),
    ("eu.anthropic.", "bedrock_converse"),
    ("anthropic.", "bedrock_converse"),
    ("meta.llama", "bedrock_converse"),
    ("amazon.", "bedrock_converse"),
    ("claude", "anthropic"),
    ("gemini", "google_genai"),
    ("mistral", "mistralai"),
    ("command", "cohere"),
    ("deepseek", "deepseek"),
)

# langchain provider token → pip package that supplies it (for error messages).
_PROVIDER_PACKAGE = {
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google_genai": "langchain-google-genai",
    "mistralai": "langchain-mistralai",
    "cohere": "langchain-cohere",
    "deepseek": "langchain-deepseek",
    "bedrock_converse": "langchain-aws",
    "bedrock": "langchain-aws",
}


def infer_provider(model: str) -> str | None:
    """Best-effort langchain ``model_provider`` for a model id (``None`` if unknown)."""
    m = (model or "").strip().lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if m.startswith(prefix):
            return provider
    return None


def resolve_research_model() -> str:
    return os.getenv("FACTOR_RESEARCH_LLM_MODEL", "gpt-4o-mini")


def resolve_research_provider(model: str | None = None) -> str | None:
    """Provider for the research LLM: explicit env wins, else inferred from the model."""
    prov = os.getenv("FACTOR_RESEARCH_LLM_PROVIDER")
    if prov:
        return prov
    return infer_provider(model or resolve_research_model())


# ════════════════════════════════════════════════════════════════════════════
# Token / cost metering
# ════════════════════════════════════════════════════════════════════════════

class LLMBudgetExceeded(RuntimeError):
    """Cumulative LLM spend crossed the ``QF_MAX_LLM_COST_USD`` hard ceiling."""


# (usd_per_1M_input_tokens, usd_per_1M_output_tokens), keyed by model-id
# substring.  Longest matching substring wins; override/extend via the
# ``QF_LLM_PRICES`` env (JSON dict model-substring -> [in, out]).
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (5.0, 30.0),
    "gpt-5.6-luna": (1.0, 6.0),
    "gpt-5.6-terra": (2.5, 15.0),
    "gpt-4o-mini": (0.15, 0.6),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "muse-spark": (1.25, 4.25),
}

# The active role for usage attribution (``None`` = fall back to the model
# instance's bound role, then "default").
llm_role: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_role", default=None)


@contextlib.contextmanager
def set_llm_role(role: str):
    """Attribute every LLM call inside the ``with`` block to ``role``."""
    token = llm_role.set(role)
    try:
        yield
    finally:
        llm_role.reset(token)


def _price_table() -> dict[str, tuple[float, float]]:
    """``DEFAULT_PRICES`` merged with the ``QF_LLM_PRICES`` env JSON overrides."""
    table = dict(DEFAULT_PRICES)
    raw = os.getenv("QF_LLM_PRICES")
    if raw:
        try:
            for key, pair in json.loads(raw).items():
                table[str(key)] = (float(pair[0]), float(pair[1]))
        except Exception:
            log.warning("QF_LLM_PRICES is not a valid JSON dict "
                        "{model-substring: [usd_in_per_1M, usd_out_per_1M]}; "
                        "ignoring it.")
    return table


_unpriced_models: set[str] = set()


def resolve_price(model: str) -> tuple[float, float] | None:
    """(usd/1M input, usd/1M output) for a model id; ``None`` if unpriced.

    Longest-substring match over the merged price table; an unknown model is
    warned about once and costs $0 (tokens are still counted).
    """
    m = (model or "").lower()
    best: tuple[float, float] | None = None
    best_len = -1
    for key, pair in _price_table().items():
        if key.lower() in m and len(key) > best_len:
            best, best_len = pair, len(key)
    if best is None and model not in _unpriced_models:
        _unpriced_models.add(model)
        log.warning("No price entry for LLM model %r — its cost is counted as "
                    "$0 (tokens still metered). Add it to QF_LLM_PRICES.", model)
    return best


class UsageMeter:
    """Thread-safe per-role accumulator of LLM calls / tokens / cost / errors."""

    _FIELDS = ("calls", "input_tokens", "output_tokens", "cost_usd", "errors")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_role: dict[str, dict[str, float]] = {}

    def _rec(self, role: str) -> dict[str, float]:
        return self._by_role.setdefault(role, {f: 0.0 for f in self._FIELDS})

    def record(self, role: str, input_tokens: int, output_tokens: int,
               cost_usd: float) -> float:
        """Add one call; returns the new cumulative total cost."""
        with self._lock:
            rec = self._rec(role)
            rec["calls"] += 1
            rec["input_tokens"] += input_tokens
            rec["output_tokens"] += output_tokens
            rec["cost_usd"] += cost_usd
            return sum(r["cost_usd"] for r in self._by_role.values())

    def record_error(self, role: str) -> None:
        with self._lock:
            self._rec(role)["errors"] += 1

    def total_cost(self) -> float:
        with self._lock:
            return sum(r["cost_usd"] for r in self._by_role.values())

    def summary(self) -> dict:
        """``{"by_role": {role: {...}}, "total": {...}}`` (ints where integral)."""
        with self._lock:
            by_role = {
                role: {f: (int(v) if f != "cost_usd" else v)
                       for f, v in rec.items()}
                for role, rec in sorted(self._by_role.items())
            }
        total = {f: 0 if f != "cost_usd" else 0.0 for f in self._FIELDS}
        for rec in by_role.values():
            for f in self._FIELDS:
                total[f] += rec[f]
        return {"by_role": by_role, "total": total}

    def reset(self) -> None:
        with self._lock:
            self._by_role.clear()


_METER = UsageMeter()


def usage_summary() -> dict:
    """Per-role + total LLM usage recorded by every metered model so far."""
    return _METER.summary()


def reset_usage() -> None:
    _METER.reset()


def budget_exhausted() -> bool:
    """True once cumulative metered cost has reached the hard ceiling.

    Cheap pre-flight check for pooled workers: a queued proposal that consults
    this before its LLM call exits instantly instead of spending money on a
    call whose completion handler would raise anyway.
    """
    ceiling = _max_cost_usd()
    return ceiling is not None and _METER.total_cost() >= ceiling


def seed_usage(summary: dict) -> None:
    """Preload the meter from a persisted ``usage_summary()`` dict.

    Called on RESUME so the per-run cost ceiling and the recorded usage are
    cumulative across process restarts instead of resetting to zero.
    """
    for role, rec in (summary.get("by_role") or {}).items():
        _METER.record(role,
                      int(rec.get("input_tokens", 0) or 0),
                      int(rec.get("output_tokens", 0) or 0),
                      float(rec.get("cost_usd", 0.0) or 0.0))


def _extract_tokens(response: Any) -> tuple[int, int]:
    """Best-effort (input, output) token counts from a LangChain ``LLMResult``.

    Mirrors the proven extraction in ``run_evolution_timing.py``: prefer the
    generations' ``message.usage_metadata`` (summed), then fall back to
    ``llm_output["token_usage"]``.
    """
    in_tok = out_tok = 0
    for gens in getattr(response, "generations", None) or []:
        for gen in gens or []:
            message = getattr(gen, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if isinstance(usage, dict) and usage:
                in_tok += int(usage.get("input_tokens", 0) or 0)
                out_tok += int(usage.get("output_tokens", 0) or 0)
    if in_tok or out_tok:
        return in_tok, out_tok
    llm_output = getattr(response, "llm_output", None) or {}
    tu = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if tu:
        return (int(tu.get("prompt_tokens", tu.get("input_tokens", 0)) or 0),
                int(tu.get("completion_tokens", tu.get("output_tokens", 0)) or 0))
    return 0, 0


def _max_cost_usd() -> float | None:
    raw = os.getenv("QF_MAX_LLM_COST_USD")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("QF_MAX_LLM_COST_USD=%r is not a float; ignoring it.", raw)
        return None


def _transcript_path() -> str | None:
    """Optional prompt/response transcript sink (``QF_LLM_TRANSCRIPT_PATH``).

    When set, every LLM call is appended as one JSON line
    ``{role, model, prompt, response, input_tokens, output_tokens}`` — the raw
    material the final-run walkthrough notebook renders. Off by default.
    """
    return os.getenv("QF_LLM_TRANSCRIPT_PATH") or None


_TRANSCRIPT_LOCK = threading.Lock()   # one JSONL line at a time across threads


def _make_usage_handler(model: str, default_role: str | None):
    """A LangChain callback handler metering one model instance into ``_METER``."""
    from langchain_core.callbacks import BaseCallbackHandler

    class _UsageHandler(BaseCallbackHandler):
        raise_error = True  # let LLMBudgetExceeded propagate out of the run

        def __init__(self) -> None:
            super().__init__()
            # Keyed by LangChain run_id so concurrent calls through the SAME
            # model instance (parallel child generation) never mismatch a
            # prompt with another call's response.
            self._prompts_by_run: dict[Any, list[str]] = {}
            self._prompts_lock = threading.Lock()

        def _role(self) -> str:
            return llm_role.get() or default_role or "default"

        def _stash(self, run_id: Any, prompts: list[str]) -> None:
            with self._prompts_lock:
                self._prompts_by_run[run_id] = prompts
                if len(self._prompts_by_run) > 64:   # never grow unbounded
                    self._prompts_by_run.pop(next(iter(self._prompts_by_run)))

        def on_llm_start(self, serialized: Any, prompts: list[str],
                         *, run_id: Any = None, **kwargs: Any) -> None:
            if _transcript_path():
                self._stash(run_id, list(prompts))

        def on_chat_model_start(self, serialized: Any, messages: Any,
                                *, run_id: Any = None, **kwargs: Any) -> None:
            if _transcript_path():
                try:
                    self._stash(run_id, [
                        "\n".join(content_text(getattr(m, "content", str(m)))
                                  for m in batch)
                        for batch in messages
                    ])
                except Exception:  # transcript is best-effort, never fatal
                    self._stash(run_id, [])

        def _write_transcript(self, response: Any, in_tok: int, out_tok: int,
                              run_id: Any) -> None:
            path = _transcript_path()
            if not path:
                return
            try:
                texts = []
                for gens in getattr(response, "generations", []) or []:
                    for g in gens:
                        msg = getattr(g, "message", None)
                        raw = getattr(msg, "content", None) or getattr(g, "text", "")
                        # thinking-enabled Anthropic content is a block LIST
                        texts.append(content_text(raw))
                with self._prompts_lock:
                    prompts = self._prompts_by_run.pop(run_id, [])
                row = {
                    "role": self._role(),
                    "model": model,
                    "prompt": "\n\n---\n\n".join(prompts),
                    "response": "\n\n---\n\n".join(t for t in texts if t),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
                from pathlib import Path
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(row) + "\n"
                with _TRANSCRIPT_LOCK, open(path, "a") as fh:
                    fh.write(line)
            except Exception:  # pragma: no cover - never break a run for logging
                log.warning("failed to append LLM transcript row", exc_info=True)

        def on_llm_end(self, response: Any, *, run_id: Any = None,
                       **kwargs: Any) -> None:
            in_tok, out_tok = _extract_tokens(response)
            price = resolve_price(model)
            cost = 0.0
            if price is not None:
                cost = (in_tok * price[0] + out_tok * price[1]) / 1e6
            self._write_transcript(response, in_tok, out_tok, run_id)
            total = _METER.record(self._role(), in_tok, out_tok, cost)
            ceiling = _max_cost_usd()
            if ceiling is not None and total > ceiling:
                raise LLMBudgetExceeded(
                    f"Cumulative LLM cost ${total:.4f} exceeded the "
                    f"QF_MAX_LLM_COST_USD ceiling ${ceiling:.2f} "
                    f"(last call: model={model!r}, role={self._role()!r}).")

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            _METER.record_error(self._role())

    return _UsageHandler()


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

# Claude models on which `temperature`/`top_p`/`top_k` return a 400 (sampling
# parameters were removed on Opus 4.7+ and the whole 5-family). Substring match
# so Bedrock ids ("anthropic.claude-opus-5", "us.anthropic....") hit too.
_NO_SAMPLING_MODELS = (
    "claude-opus-5", "claude-fable-5", "claude-mythos-5", "claude-sonnet-5",
    "claude-opus-4-7", "claude-opus-4-8",
)


def _rejects_sampling_params(model: str) -> bool:
    m = (model or "").lower()
    return any(s in m for s in _NO_SAMPLING_MODELS)


def content_text(content: Any) -> str:
    """Flatten a chat-message ``content`` to plain text.

    Anthropic models with thinking enabled return ``content`` as a LIST of
    typed blocks (thinking + text); every consumer in this codebase expects a
    plain string.  Text blocks are concatenated; thinking/other blocks are
    dropped (their tokens are still metered).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


#: Outage tolerance: connection-level failures retry with capped exponential
#: backoff for up to ~30 minutes TOTAL per call, so a WiFi drop STALLS a run
#: instead of starving whole generations (a starved generation still burns its
#: progressive-reveal window). Overridable via env for tests.
_CONN_RETRY_SCHEDULE = (15, 30, 60, 120, 300, 300, 300, 300, 300)


def _is_connection_error(e: BaseException) -> bool:
    """Best-effort: transport/DNS/timeouts — NOT HTTP 4xx/5xx or budget stops."""
    if isinstance(e, LLMBudgetExceeded):
        return False
    names = {type(x).__name__ for x in (e, e.__cause__, e.__context__) if x}
    markers = ("APIConnectionError", "APITimeoutError", "ConnectError",
               "ConnectTimeout", "ReadTimeout", "ReadError", "WriteError",
               "PoolTimeout", "RemoteProtocolError", "ConnectionError",
               "TimeoutException", "gaierror", "TransportError")
    return any(any(m in n for m in markers) for n in names)


class _TextContentModel:
    """Proxy for Anthropic-family chat models with two behaviours:

    1. flattens list-content responses to plain-text ``content`` (the contract
       every ``_parse_json`` call site in the agents assumes);
    2. persistent connection-error retries with capped backoff — an internet
       outage pauses the call for up to ~30 min instead of failing it.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def invoke(self, *args: Any, **kwargs: Any):
        attempt = 0
        while True:
            try:
                resp = self._inner.invoke(*args, **kwargs)
                break
            except Exception as e:  # noqa: BLE001 — classify then re-raise
                if not _is_connection_error(e) or attempt >= len(_CONN_RETRY_SCHEDULE):
                    raise
                delay = _CONN_RETRY_SCHEDULE[attempt]
                attempt += 1
                log.warning(
                    "connection problem on LLM call (%s: %s) — waiting %ds and "
                    "retrying (outage retry %d/%d)", type(e).__name__, e, delay,
                    attempt, len(_CONN_RETRY_SCHEDULE))
                time.sleep(delay)
        try:
            if not isinstance(getattr(resp, "content", None), str):
                resp.content = content_text(resp.content)
        except Exception:  # noqa: BLE001 — never break the response path
            pass
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def make_chat_llm(
    model: str | None = None,
    provider: str | None = None,
    *,
    temperature: float = 0.6,
    timeout: float | None = 120,
    max_retries: int = 4,
    base_url: str | None = None,
    role: str | None = None,
    **kwargs: Any,
):
    """Build a langchain chat model for any supported provider.

    ``model`` defaults to ``FACTOR_RESEARCH_LLM_MODEL`` (then ``gpt-4o-mini``);
    ``provider`` defaults to ``FACTOR_RESEARCH_LLM_PROVIDER`` or is inferred from
    the model id.  Finite ``timeout`` + ``max_retries`` mirror the agents' direct
    ``ChatOpenAI`` construction so a stalled response can't hang the pipeline.

    ``base_url`` (or the ``FACTOR_RESEARCH_LLM_BASE_URL`` env, which it wins
    over) points the OpenAI provider at any OpenAI-compatible endpoint;
    ``role`` binds a default usage-attribution role for this instance (see
    :func:`set_llm_role` / :func:`usage_summary`).

    Raises a clear, actionable error if the provider's integration package isn't
    installed (e.g. asks the user to ``pip install langchain-anthropic``).
    """
    from langchain.chat_models import init_chat_model

    model = model or resolve_research_model()
    provider = provider or resolve_research_provider(model)
    effective_base_url = base_url or os.getenv("FACTOR_RESEARCH_LLM_BASE_URL")
    if effective_base_url and (provider or infer_provider(model)) == "openai":
        kwargs["base_url"] = effective_base_url
    callbacks = list(kwargs.pop("callbacks", None) or [])
    callbacks.append(_make_usage_handler(model, role))
    # Long-thinking Claude calls routinely exceed 2 minutes; a 120s client
    # timeout aborts and retries them forever (the aborted attempt may still
    # be billed server-side). Floor the timeout for Anthropic-family providers
    # — this also overrides call sites that hard-code timeout=120.
    if (provider or infer_provider(model)) in (
            "anthropic", "bedrock_converse", "bedrock"):
        if timeout is not None and timeout < 600:
            timeout = 600
    # Claude 4.7+/5-family models REJECT sampling parameters with a 400
    # ("`temperature` is deprecated for this model") — never send them there.
    sampling: dict[str, Any] = {}
    if not _rejects_sampling_params(model):
        sampling["temperature"] = temperature
    else:
        for k in ("top_p", "top_k"):
            kwargs.pop(k, None)
    try:
        chat = init_chat_model(
            model,
            model_provider=provider,
            timeout=timeout,
            max_retries=max_retries,
            callbacks=callbacks,
            **sampling,
            **kwargs,
        )
        if (provider or infer_provider(model)) in (
                "anthropic", "bedrock_converse", "bedrock"):
            chat = _TextContentModel(chat)
        return chat
    except ImportError as e:
        pkg = _PROVIDER_PACKAGE.get(provider or "", f"langchain-{provider}")
        raise ImportError(
            f"Chat model {model!r} (provider={provider!r}) needs the "
            f"'{pkg}' integration package. Install it with "
            f"`./venv/bin/pip install {pkg}` and set its API key in .env."
        ) from e
