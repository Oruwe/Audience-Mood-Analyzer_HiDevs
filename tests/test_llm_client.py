"""Tests for engine.llm_client's provider failover and model-chain routing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from engine.llm_client import _MODEL_CHAIN, analyze_comment, flush_observability
from schemas import RawComment


def _analysis_json(**overrides) -> str:
    payload = {
        "sentiment": "positive",
        "confidence": 0.9,
        "primary_intent": "praise_endorsement",
        "urgency_score": 0.1,
        "emotional_drivers": ["joy"],
        "summary": "Nice.",
        "recommended_action": "amplify_marketing",
        "suggested_reply_draft": None,
        "brand_safety_flag": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def _fake_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _comment() -> RawComment:
    return RawComment(
        id="c1", platform="twitter", text="Loving this!",
        author_handle="@fan", timestamp=datetime.now(UTC),
    )


async def test_no_keys_set_raises_without_attempting_any_call(monkeypatch):
    calls: list[str] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return _fake_response(_analysis_json())

    monkeypatch.setattr("engine.llm_client.acompletion", fake_acompletion)
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await analyze_comment(_comment())
    assert calls == []  # every chain entry was skipped for lacking a key


async def test_first_provider_success_short_circuits(monkeypatch):
    primary_model, primary_key_env = _MODEL_CHAIN[0]
    monkeypatch.setenv(primary_key_env, "fake-key")
    calls: list[str] = []

    async def fake_acompletion(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return _fake_response(_analysis_json())

    monkeypatch.setattr("engine.llm_client.acompletion", fake_acompletion)
    record = await analyze_comment(_comment())

    assert calls == [primary_model]
    assert record.model_used == primary_model
    assert record.comment_id == "c1"
    assert record.sentiment.value == "positive"


async def test_falls_back_to_second_provider_on_failure(monkeypatch):
    for _, key_env in _MODEL_CHAIN:
        monkeypatch.setenv(key_env, "fake-key")
    primary_model, _ = _MODEL_CHAIN[0]
    fallback_model, _ = _MODEL_CHAIN[1]
    calls: list[str] = []

    async def fake_acompletion(*args, **kwargs):
        model = kwargs.get("model")
        calls.append(model)
        if model == primary_model:
            raise RuntimeError("simulated provider outage")
        return _fake_response(_analysis_json())

    monkeypatch.setattr("engine.llm_client.acompletion", fake_acompletion)
    record = await analyze_comment(_comment())

    assert calls == [primary_model, fallback_model]
    assert record.model_used == fallback_model


async def test_all_providers_failing_raises_with_last_error(monkeypatch):
    for _, key_env in _MODEL_CHAIN:
        monkeypatch.setenv(key_env, "fake-key")
    last_model = _MODEL_CHAIN[-1][0]

    async def fake_acompletion(*args, **kwargs):
        raise RuntimeError(f"outage:{kwargs.get('model')}")

    monkeypatch.setattr("engine.llm_client.acompletion", fake_acompletion)
    with pytest.raises(RuntimeError, match=f"outage:{last_model}"):
        await analyze_comment(_comment())


def test_flush_observability_is_a_safe_noop_without_langfuse_keys():
    flush_observability()  # must not raise even though Langfuse isn't configured
