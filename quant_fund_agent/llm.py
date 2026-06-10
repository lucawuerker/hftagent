"""Provider-agnostic chat-LLM factory.

The agents historically each constructed ``langchain_openai.ChatOpenAI`` directly,
hard-wiring OpenAI.  To compare *different* research models — including non-OpenAI
ones — in the factor-creation phase, the factor researcher builds its LLM through
:func:`make_chat_llm`, which wraps langchain's multi-provider
``init_chat_model`` and infers the provider from the model id when not given.

Only the factor researcher is wired to this today; the factory is intentionally
generic so the other agents can adopt it later.

Non-OpenAI providers need their langchain integration package installed
(``langchain-anthropic``, ``langchain-google-genai``, …); :func:`make_chat_llm`
raises a clear, actionable error if it is missing.
"""

from __future__ import annotations

import os
from typing import Any

# Model-id prefix → langchain ``model_provider`` token.  Checked in order.
_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt", "openai"),
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


def make_chat_llm(
    model: str | None = None,
    provider: str | None = None,
    *,
    temperature: float = 0.6,
    timeout: float | None = 120,
    max_retries: int = 4,
    **kwargs: Any,
):
    """Build a langchain chat model for any supported provider.

    ``model`` defaults to ``FACTOR_RESEARCH_LLM_MODEL`` (then ``gpt-4o-mini``);
    ``provider`` defaults to ``FACTOR_RESEARCH_LLM_PROVIDER`` or is inferred from
    the model id.  Finite ``timeout`` + ``max_retries`` mirror the agents' direct
    ``ChatOpenAI`` construction so a stalled response can't hang the pipeline.

    Raises a clear, actionable error if the provider's integration package isn't
    installed (e.g. asks the user to ``pip install langchain-anthropic``).
    """
    from langchain.chat_models import init_chat_model

    model = model or resolve_research_model()
    provider = provider or resolve_research_provider(model)
    try:
        return init_chat_model(
            model,
            model_provider=provider,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    except ImportError as e:
        pkg = _PROVIDER_PACKAGE.get(provider or "", f"langchain-{provider}")
        raise ImportError(
            f"Chat model {model!r} (provider={provider!r}) needs the "
            f"'{pkg}' integration package. Install it with "
            f"`./venv/bin/pip install {pkg}` and set its API key in .env."
        ) from e
