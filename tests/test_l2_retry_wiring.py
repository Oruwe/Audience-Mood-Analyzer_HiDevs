"""L2 contract tests — SPEC §8 retry wiring on the OpenRouter call sites
(engine.batching's chat completion call, engine.stage_a's embeddings call).

Only the exception-classification + retry-until-success behavior;
resilience.retry_transient's own mechanics are covered by
tests/test_l1_resilience.py.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from litellm.exceptions import AuthenticationError, ServiceUnavailableError

import engine.batching as batching
import engine.stage_a as stage_a
from schemas import RawComment, StageASentimentBatch

API_KEY = "fake-key"


def _comment(i: int) -> RawComment:
    return RawComment(
        id=f"c{i}", platform="youtube", text=f"comment {i}",
        timestamp=datetime.now(timezone.utc), video_id="v1",
    )


def test_batching_retries_a_transient_litellm_error(monkeypatch):
    comments = [_comment(0)]
    valid_json = json.dumps({
        "results": [{"comment_id": "c0", "sentiment": "positive", "confidence": 0.9}]
    })
    calls = {"count": 0}

    async def flaky_acompletion(**kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise ServiceUnavailableError(
                message="try again", model="m", llm_provider="openrouter"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(batching, "acompletion", flaky_acompletion)
    result = asyncio.run(batching.run_batched_llm_classification(
        comments, api_key=API_KEY, model="openrouter/x", system_prompt="sys",
        response_schema=StageASentimentBatch, stage_label="test",
    ))

    assert calls["count"] == 2
    assert result["c0"].sentiment.value == "positive"


def test_batching_does_not_retry_a_non_transient_error(monkeypatch):
    comments = [_comment(0)]
    calls = {"count": 0}

    async def always_unauthenticated(**kwargs):
        calls["count"] += 1
        raise AuthenticationError(message="bad key", model="m", llm_provider="openrouter")

    monkeypatch.setattr(batching, "acompletion", always_unauthenticated)

    with pytest.raises(AuthenticationError):
        asyncio.run(batching.run_batched_llm_classification(
            comments, api_key=API_KEY, model="openrouter/x", system_prompt="sys",
            response_schema=StageASentimentBatch, stage_label="test",
        ))
    assert calls["count"] == 1  # not a retryable type -- fails on the first attempt


def test_embeddings_retries_a_503_until_success():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await stage_a.embed_comments_batch(
                [_comment(0)], client=client, api_key=API_KEY
            )

    result = asyncio.run(scenario())
    assert result == {"c0": [1.0]}
    assert calls["count"] == 3


def test_embeddings_does_not_retry_a_401():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(401, text="bad key")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await stage_a.embed_comments_batch(
                [_comment(0)], client=client, api_key=API_KEY
            )

    with pytest.raises(stage_a.EmbeddingAPIError):
        asyncio.run(scenario())
    assert calls["count"] == 1
