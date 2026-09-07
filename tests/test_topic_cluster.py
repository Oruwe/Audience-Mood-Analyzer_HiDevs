"""Tests for engine.topic_cluster.extract_trending_theme."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import litellm

from engine.topic_cluster import _FALLBACK_THEME, _MIN_CLUSTER_SIZE, extract_trending_theme
from schemas import EnrichedCommentRecord, PrimaryIntent, RecommendedAction, Sentiment


def _record(i: int, summary: str, embedding: list[float] | None = None) -> EnrichedCommentRecord:
    return EnrichedCommentRecord(
        comment_id=f"c{i}", platform="bluesky", author_handle=f"@u{i}",
        raw_text=f"text {i}", sentiment=Sentiment.NEGATIVE, confidence=0.9,
        primary_intent=PrimaryIntent.BUG_REPORT, urgency_score=0.9,
        emotional_drivers=["frustration"], summary=summary,
        recommended_action=RecommendedAction.ESCALATE_TO_SUPPORT,
        suggested_reply_draft=None, brand_safety_flag=False,
        embedding=embedding, cluster_id=None, latency_ms=10.0,
        model_used="hermetic", processed_at=datetime.now(UTC),
    )


def _stub_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


async def test_below_min_cluster_size_returns_fallback_without_llm_call(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(*args, **kwargs):
        calls["n"] += 1
        return _stub_response("x")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    comments = [_record(i, f"s{i}") for i in range(_MIN_CLUSTER_SIZE - 1)]
    theme = await extract_trending_theme(comments)

    assert theme == _FALLBACK_THEME
    assert calls["n"] == 0


async def test_missing_api_key_returns_fallback_without_llm_call(monkeypatch):
    calls = {"n": 0}

    async def fake_acompletion(*args, **kwargs):
        calls["n"] += 1
        return _stub_response("x")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    comments = [_record(i, f"s{i}") for i in range(_MIN_CLUSTER_SIZE)]
    theme = await extract_trending_theme(comments)

    assert theme == _FALLBACK_THEME
    assert calls["n"] == 0


async def test_happy_path_strips_quotes_from_llm_output(monkeypatch):
    async def fake_acompletion(*args, **kwargs):
        return _stub_response('  "Login Outage Storm"  ')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    comments = [_record(i, f"s{i}") for i in range(_MIN_CLUSTER_SIZE)]
    theme = await extract_trending_theme(comments)

    assert theme == "Login Outage Storm"


async def test_llm_failure_falls_back(monkeypatch):
    async def fake_acompletion(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    comments = [_record(i, f"s{i}") for i in range(_MIN_CLUSTER_SIZE)]
    theme = await extract_trending_theme(comments)

    assert theme == _FALLBACK_THEME


async def test_uses_embeddings_when_enough_vectors_present(monkeypatch):
    """Regression guard: the KMeans branch must still reach the LLM call."""
    seen_prompts: list[str] = []

    async def fake_acompletion(*args, **kwargs):
        seen_prompts.append(kwargs["messages"][0]["content"])
        return _stub_response("Some Theme")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    comments = [_record(i, f"summary-{i}", embedding=[float(i)] * 4) for i in range(5)]
    theme = await extract_trending_theme(comments)

    assert theme == "Some Theme"
    assert seen_prompts and "summary-" in seen_prompts[0]
