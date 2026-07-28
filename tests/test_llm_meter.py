"""Unit tests for the LLM usage meter, pricing, budget ceiling and the
Bedrock/base-url wiring in ``quant_fund_agent.llm``.  No network calls: token
shapes are faked with real langchain-core message/result objects and
``init_chat_model`` is stubbed where construction kwargs matter.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from quant_fund_agent.llm import (
    DEFAULT_PRICES,
    LLMBudgetExceeded,
    _make_usage_handler,
    infer_provider,
    make_chat_llm,
    reset_usage,
    resolve_price,
    set_llm_role,
    usage_summary,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _result_usage_metadata(in_tok: int = 100, out_tok: int = 40) -> LLMResult:
    """LLMResult carrying tokens on the message's ``usage_metadata``."""
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": in_tok, "output_tokens": out_tok,
                        "total_tokens": in_tok + out_tok},
    )
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def _result_token_usage(prompt: int = 70, completion: int = 30) -> LLMResult:
    """LLMResult carrying tokens only on ``llm_output['token_usage']``."""
    msg = AIMessage(content="hi")
    return LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"token_usage": {"prompt_tokens": prompt,
                                    "completion_tokens": completion}},
    )


@pytest.fixture(autouse=True)
def _clean_meter(monkeypatch):
    monkeypatch.delenv("QF_MAX_LLM_COST_USD", raising=False)
    monkeypatch.delenv("QF_LLM_PRICES", raising=False)
    reset_usage()
    yield
    reset_usage()


# ── token extraction ────────────────────────────────────────────────────────

def test_tokens_from_usage_metadata():
    handler = _make_usage_handler("gpt-4o-mini", None)
    handler.on_llm_end(_result_usage_metadata(100, 40))
    total = usage_summary()["total"]
    assert total["calls"] == 1
    assert total["input_tokens"] == 100
    assert total["output_tokens"] == 40


def test_tokens_from_llm_output_token_usage():
    handler = _make_usage_handler("gpt-4o-mini", None)
    handler.on_llm_end(_result_token_usage(70, 30))
    total = usage_summary()["total"]
    assert total["input_tokens"] == 70
    assert total["output_tokens"] == 30


def test_tokenless_result_still_counts_the_call():
    handler = _make_usage_handler("gpt-4o-mini", None)
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(
        message=AIMessage(content="hi"))]]))
    total = usage_summary()["total"]
    assert total["calls"] == 1
    assert total["input_tokens"] == 0
    assert total["cost_usd"] == 0.0


# ── role attribution ────────────────────────────────────────────────────────

def test_role_attribution_context_var_and_instance_default():
    bound = _make_usage_handler("gpt-4o-mini", "codegen")
    unbound = _make_usage_handler("gpt-4o-mini", None)
    unbound.on_llm_end(_result_usage_metadata())          # → "default"
    bound.on_llm_end(_result_usage_metadata())            # → "codegen"
    with set_llm_role("hypothesis"):                      # ctx var wins
        bound.on_llm_end(_result_usage_metadata())
    by_role = usage_summary()["by_role"]
    assert by_role["default"]["calls"] == 1
    assert by_role["codegen"]["calls"] == 1
    assert by_role["hypothesis"]["calls"] == 1


def test_role_resets_after_context_exit():
    handler = _make_usage_handler("gpt-4o-mini", None)
    with set_llm_role("debate"):
        pass
    handler.on_llm_end(_result_usage_metadata())
    assert "debate" not in usage_summary()["by_role"]


def test_error_attribution():
    handler = _make_usage_handler("gpt-4o-mini", "codegen")
    handler.on_llm_error(RuntimeError("boom"))
    by_role = usage_summary()["by_role"]
    assert by_role["codegen"]["errors"] == 1
    assert by_role["codegen"]["calls"] == 0


# ── pricing ─────────────────────────────────────────────────────────────────

def test_default_price_table_lookup():
    assert resolve_price("gpt-4o-mini") == DEFAULT_PRICES["gpt-4o-mini"]
    # substring match: provider-decorated ids still resolve
    assert resolve_price("anthropic.claude-opus-5-v1:0") == \
        DEFAULT_PRICES["claude-opus-5"]


def test_longest_substring_wins(monkeypatch):
    monkeypatch.setenv("QF_LLM_PRICES", json.dumps({"gpt-5.6": [9.0, 9.0]}))
    # "gpt-5.6-luna" (len 12) beats the override "gpt-5.6" (len 7)
    assert resolve_price("gpt-5.6-luna") == (1.0, 6.0)
    assert resolve_price("gpt-5.6-nova") == (9.0, 9.0)


def test_env_price_override(monkeypatch):
    monkeypatch.setenv("QF_LLM_PRICES",
                       json.dumps({"gpt-4o-mini": [100.0, 200.0]}))
    assert resolve_price("gpt-4o-mini") == (100.0, 200.0)


def test_unknown_model_costs_zero_but_counts_tokens():
    handler = _make_usage_handler("totally-unknown-model", None)
    handler.on_llm_end(_result_usage_metadata(50, 50))
    total = usage_summary()["total"]
    assert total["cost_usd"] == 0.0
    assert total["input_tokens"] == 50


def test_cost_arithmetic():
    handler = _make_usage_handler("gpt-4o-mini", None)  # (0.15, 0.6) per 1M
    handler.on_llm_end(_result_usage_metadata(1_000_000, 1_000_000))
    assert usage_summary()["total"]["cost_usd"] == pytest.approx(0.15 + 0.6)


# ── budget ceiling ──────────────────────────────────────────────────────────

def test_budget_ceiling_raises(monkeypatch):
    monkeypatch.setenv("QF_MAX_LLM_COST_USD", "0.0001")
    handler = _make_usage_handler("gpt-4o-mini", None)
    with pytest.raises(LLMBudgetExceeded):
        handler.on_llm_end(_result_usage_metadata(1_000_000, 1_000_000))
    # the call itself is still recorded before the raise
    assert usage_summary()["total"]["calls"] == 1


def test_no_ceiling_when_env_unset():
    handler = _make_usage_handler("gpt-4o-mini", None)
    handler.on_llm_end(_result_usage_metadata(1_000_000, 1_000_000))  # no raise


# ── summary structure ───────────────────────────────────────────────────────

def test_usage_summary_structure_and_reset():
    handler = _make_usage_handler("gpt-4o-mini", "codegen")
    handler.on_llm_end(_result_usage_metadata(10, 5))
    s = usage_summary()
    assert set(s) == {"by_role", "total"}
    for rec in [s["total"], *s["by_role"].values()]:
        assert set(rec) == {"calls", "input_tokens", "output_tokens",
                            "cost_usd", "errors"}
    assert s["total"]["input_tokens"] == 10
    reset_usage()
    assert usage_summary() == {"by_role": {}, "total": {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cost_usd": 0.0, "errors": 0}}


# ── provider inference (Bedrock) ────────────────────────────────────────────

@pytest.mark.parametrize("model,provider", [
    ("anthropic.claude-opus-5", "bedrock_converse"),
    ("us.anthropic.claude-sonnet-5-v1:0", "bedrock_converse"),
    ("eu.anthropic.claude-fable-5", "bedrock_converse"),
    ("meta.llama4-maverick", "bedrock_converse"),
    ("amazon.nova-pro-v1:0", "bedrock_converse"),
    ("claude-opus-5", "anthropic"),   # bare claude still → Anthropic API
    ("gpt-4o-mini", "openai"),
])
def test_infer_provider(model, provider):
    assert infer_provider(model) == provider


# ── base_url passthrough ────────────────────────────────────────────────────

def _stub_init_chat_model(monkeypatch):
    captured = {}

    def stub(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain.chat_models.init_chat_model", stub)
    return captured


def test_base_url_env_passthrough_openai(monkeypatch):
    captured = _stub_init_chat_model(monkeypatch)
    monkeypatch.setenv("FACTOR_RESEARCH_LLM_BASE_URL", "https://models.example/v1")
    monkeypatch.delenv("FACTOR_RESEARCH_LLM_PROVIDER", raising=False)
    make_chat_llm("gpt-4o-mini")
    assert captured["base_url"] == "https://models.example/v1"
    assert captured["model_provider"] == "openai"


def test_base_url_kwarg_wins_over_env(monkeypatch):
    captured = _stub_init_chat_model(monkeypatch)
    monkeypatch.setenv("FACTOR_RESEARCH_LLM_BASE_URL", "https://env.example/v1")
    make_chat_llm("gpt-4o-mini", base_url="https://kwarg.example/v1")
    assert captured["base_url"] == "https://kwarg.example/v1"


def test_base_url_not_passed_for_non_openai(monkeypatch):
    captured = _stub_init_chat_model(monkeypatch)
    monkeypatch.setenv("FACTOR_RESEARCH_LLM_BASE_URL", "https://models.example/v1")
    monkeypatch.delenv("FACTOR_RESEARCH_LLM_PROVIDER", raising=False)
    make_chat_llm("anthropic.claude-opus-5")
    assert "base_url" not in captured
    assert captured["model_provider"] == "bedrock_converse"


def test_make_chat_llm_attaches_usage_callback(monkeypatch):
    captured = _stub_init_chat_model(monkeypatch)
    monkeypatch.delenv("FACTOR_RESEARCH_LLM_BASE_URL", raising=False)
    make_chat_llm("gpt-4o-mini", role="codegen")
    callbacks = captured["callbacks"]
    assert len(callbacks) == 1
    callbacks[0].on_llm_end(_result_usage_metadata(10, 5))
    assert usage_summary()["by_role"]["codegen"]["calls"] == 1
