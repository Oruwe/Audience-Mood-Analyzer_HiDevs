"""L2 contract tests — SPEC §8 retry wiring on the OpenRouter call sites
(engine.batching's chat completion call, engine.stage_a's embeddings call,
engine.insights's Stage C synthesis call).

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
from litellm.exceptions import APIError, AuthenticationError, NotFoundError, RateLimitError, ServiceUnavailableError

import engine.batching as batching
import engine.insights as insights
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


def _bare_api_error(model: str) -> APIError:
    """Live incident (2026-09-13): a real "Nvidia: Service temporarily
    overloaded" failure surfaced from litellm as the bare base APIError,
    not one of the more specific subclasses (ServiceUnavailableError etc.)
    this project used to retry on by name -- OpenRouter doesn't always wrap
    a transient upstream failure into a specific subclass."""
    return APIError(status_code=500, message="Service temporarily overloaded",
                     llm_provider="openrouter", model=model)


def test_batching_retries_a_bare_api_error_not_just_named_subclasses(monkeypatch):
    comments = [_comment(0)]
    valid_json = json.dumps({
        "results": [{"comment_id": "c0", "sentiment": "positive", "confidence": 0.9}]
    })
    calls = {"count": 0}

    async def flaky_acompletion(*, model, **kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise _bare_api_error(model)
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


def test_batching_falls_back_on_a_bare_api_error_from_the_primary(monkeypatch):
    calls = {"model-a": 0, "model-b": 0}
    valid_json = json.dumps({
        "results": [{"comment_id": "c0", "sentiment": "positive", "confidence": 0.9}]
    })

    async def fake_acompletion(*, model, **kwargs):
        calls[model] += 1
        if model == "model-a":
            raise _bare_api_error(model)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(batching.run_batched_llm_classification(
        [_comment(0)], api_key=API_KEY, model="model-a", system_prompt="sys",
        response_schema=StageASentimentBatch, stage_label="test",
        fallback_models=("model-b",),
    ))

    assert calls["model-a"] == 4  # exhausted every retry before falling back
    assert calls["model-b"] == 1
    assert result["c0"].sentiment.value == "positive"


def test_batching_falls_back_immediately_on_a_not_found_error_without_retrying_it(monkeypatch):
    """Live incident (2026-09-13): OpenRouter 404'd a withdrawn `:free`
    slug -- "This model is unavailable for free ... use this slug
    instead: ...". Retrying the same model is pointless (it's gone), but
    falling back to a different one is exactly right, and should happen
    on the FIRST attempt, not after retry_transient_api_error's usual
    multi-attempt budget (NotFoundError isn't retryable, just
    fallback-worthy -- see resilience.py).
    """
    calls = {"model-a": 0, "model-b": 0}
    valid_json = json.dumps({
        "results": [{"comment_id": "c0", "sentiment": "positive", "confidence": 0.9}]
    })

    async def fake_acompletion(*, model, **kwargs):
        calls[model] += 1
        if model == "model-a":
            raise NotFoundError(
                message="This model is unavailable for free ... use a different slug",
                model=model, llm_provider="openrouter",
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(batching.run_batched_llm_classification(
        [_comment(0)], api_key=API_KEY, model="model-a", system_prompt="sys",
        response_schema=StageASentimentBatch, stage_label="test",
        fallback_models=("model-b",),
    ))

    assert calls["model-a"] == 1  # not retried -- retrying a withdrawn model is pointless
    assert calls["model-b"] == 1
    assert result["c0"].sentiment.value == "positive"


def test_insights_stage_c_falls_back_immediately_on_a_not_found_error(monkeypatch):
    valid_json = json.dumps({
        "theme": "Wants a Docker follow-up",
        "quotes": ["a real quote", "another real quote"],
        "suggested_title": "Docker 101",
    })
    calls = {"model-a": 0, "model-b": 0}

    async def fake_acompletion(*, model, **kwargs):
        calls[model] += 1
        if model == "model-a":
            raise NotFoundError(
                message="This model is unavailable for free ... use a different slug",
                model=model, llm_provider="openrouter",
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)
    from schemas import RequestInsightDraft

    result = asyncio.run(insights._call_stage_c_with_fallback(
        ("model-a", "model-b"), API_KEY, [{"role": "user", "content": "hi"}], RequestInsightDraft,
    ))

    assert calls["model-a"] == 1
    assert calls["model-b"] == 1
    assert result.choices[0].message.content == valid_json


def test_batching_falls_back_immediately_on_a_rate_limit_without_retrying_it(monkeypatch):
    """Live incident (2026-09-13, config/models.py's *_FALLBACK constants):
    a free OpenRouter model's 429 is a *sustained* shared-pool exhaustion,
    not a momentary blip -- confirmed by watching a real job hit the same
    429 on the same model dozens of times over several minutes. Retrying
    the SAME model (resilience.py's old behavior: up to 4 attempts,
    backoff to 8s) before falling back was pure wasted latency once that
    was known, so RateLimitError now skips straight to the fallback model
    on the first attempt -- same as the NotFoundError case above, for a
    latency reason rather than a "this model doesn't exist" one.
    """
    comments = [_comment(0)]
    valid_json = json.dumps({
        "results": [{"comment_id": "c0", "sentiment": "positive", "confidence": 0.9}]
    })
    calls = {"model-a": 0, "model-b": 0}

    async def fake_acompletion(*, model, **kwargs):
        calls[model] += 1
        if model == "model-a":
            raise RateLimitError(message="rate limited", model=model, llm_provider="openrouter")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(batching.run_batched_llm_classification(
        comments, api_key=API_KEY, model="model-a", system_prompt="sys",
        response_schema=StageASentimentBatch, stage_label="test",
        fallback_models=("model-b",),
    ))

    assert calls["model-a"] == 1  # not retried -- a sustained 429 wouldn't clear in time anyway
    assert calls["model-b"] == 1
    assert result["c0"].sentiment.value == "positive"


def test_batching_raises_when_every_model_including_fallbacks_is_exhausted(monkeypatch):
    comments = [_comment(0)]

    async def always_rate_limited(*, model, **kwargs):
        raise RateLimitError(message="rate limited", model=model, llm_provider="openrouter")

    monkeypatch.setattr(batching, "acompletion", always_rate_limited)

    with pytest.raises(RateLimitError):
        asyncio.run(batching.run_batched_llm_classification(
            comments, api_key=API_KEY, model="model-a", system_prompt="sys",
            response_schema=StageASentimentBatch, stage_label="test",
            fallback_models=("model-b",),
        ))


def test_insights_stage_c_falls_back_immediately_on_a_rate_limit_without_retrying_it(monkeypatch):
    valid_json = json.dumps({
        "theme": "Wants a Docker follow-up",
        "quotes": ["a real quote", "another real quote"],
        "suggested_title": "Docker 101",
    })
    calls = {"model-a": 0, "model-b": 0}

    async def fake_acompletion(*, model, **kwargs):
        calls[model] += 1
        if model == "model-a":
            raise RateLimitError(message="rate limited", model=model, llm_provider="openrouter")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)
    from schemas import RequestInsightDraft

    result = asyncio.run(insights._call_stage_c_with_fallback(
        ("model-a", "model-b"), API_KEY, [{"role": "user", "content": "hi"}], RequestInsightDraft,
    ))

    assert calls["model-a"] == 1  # not retried -- see the batching.py test's docstring
    assert calls["model-b"] == 1
    assert result.choices[0].message.content == valid_json


def test_insights_stage_c_retries_a_transient_litellm_error(monkeypatch):
    valid_json = json.dumps({
        "theme": "Wants a Docker follow-up",
        "quotes": ["a real quote", "another real quote"],
        "suggested_title": "Docker 101",
    })
    calls = {"count": 0}

    async def flaky_acompletion(**kwargs):
        calls["count"] += 1
        if calls["count"] < 2:
            raise ServiceUnavailableError(message="try again", model="m", llm_provider="openrouter")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=valid_json))]
        )

    monkeypatch.setattr(insights, "acompletion", flaky_acompletion)
    from schemas import RequestInsightDraft

    result = asyncio.run(insights._call_stage_c(
        "openrouter/x", API_KEY, [{"role": "user", "content": "hi"}], RequestInsightDraft,
    ))

    assert calls["count"] == 2
    assert result.choices[0].message.content == valid_json


def test_insights_stage_c_does_not_retry_a_non_transient_error(monkeypatch):
    calls = {"count": 0}

    async def always_unauthenticated(**kwargs):
        calls["count"] += 1
        raise AuthenticationError(message="bad key", model="m", llm_provider="openrouter")

    monkeypatch.setattr(insights, "acompletion", always_unauthenticated)
    from schemas import RequestInsightDraft

    with pytest.raises(AuthenticationError):
        asyncio.run(insights._call_stage_c(
            "openrouter/x", API_KEY, [{"role": "user", "content": "hi"}], RequestInsightDraft,
        ))
    assert calls["count"] == 1
