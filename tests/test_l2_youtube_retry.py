"""L2 contract tests — ingestion.youtube's SPEC §8 retry wiring on `_get`.

Only checks the exception-classification + retry-until-success behavior;
resilience.retry_transient's own mechanics are covered by
tests/test_l1_resilience.py.
"""

import asyncio

import httpx
import pytest

from ingestion.youtube import YouTubeAPIError, _get


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_503_is_retried_until_it_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, text="service unavailable")
        return httpx.Response(200, json={"items": []})

    async def scenario():
        async with _client(handler) as client:
            return await _get(client, "channels", {"id": "UCabc"}, "key")

    result = asyncio.run(scenario())
    assert result == {"items": []}
    assert calls["count"] == 3


def test_429_is_retried_too():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 2:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"items": []})

    async def scenario():
        async with _client(handler) as client:
            return await _get(client, "channels", {"id": "UCabc"}, "key")

    asyncio.run(scenario())
    assert calls["count"] == 2


def test_403_is_never_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(403, text="forbidden")

    async def scenario():
        async with _client(handler) as client:
            return await _get(client, "commentThreads", {"videoId": "v1"}, "key")

    with pytest.raises(YouTubeAPIError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 403
    assert calls["count"] == 1  # not retried


def test_400_is_never_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, text="bad request")

    async def scenario():
        async with _client(handler) as client:
            return await _get(client, "channels", {}, "key")

    with pytest.raises(YouTubeAPIError):
        asyncio.run(scenario())
    assert calls["count"] == 1
