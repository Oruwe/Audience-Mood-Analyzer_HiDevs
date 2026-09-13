"""L2 contract tests — engine.stage_a embeddings, against a stubbed
OpenRouter /embeddings endpoint (httpx.MockTransport, no network, no key).

SPEC §4.1 explicitly deletes V2's hash-vector embedding fallback because it
"silently made clustering meaningless whenever it fired" -- these tests
assert the replacement raises cleanly instead of ever inventing a vector.
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

import engine.stage_a as stage_a
from schemas import RawComment

API_KEY = "fake-openrouter-key"


def _comment(i: int) -> RawComment:
    return RawComment(
        id=f"c{i}", platform="youtube", text=f"comment number {i}",
        timestamp=datetime.now(timezone.utc), video_id="v1",
    )


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_empty_batch_short_circuits_without_a_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make a request for an empty batch")

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch([], client=client, api_key=API_KEY)

    assert asyncio.run(scenario()) == {}


def test_happy_path_maps_embeddings_back_by_comment_id_even_out_of_order():
    comments = [_comment(i) for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        # Respond with the items in a shuffled index order, as a real API
        # response is not guaranteed to preserve input order.
        return httpx.Response(200, json={"data": [
            {"index": 2, "embedding": [2.0, 2.0]},
            {"index": 0, "embedding": [0.0, 0.0]},
            {"index": 1, "embedding": [1.0, 1.0]},
        ]})

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(comments, client=client, api_key=API_KEY)

    result = asyncio.run(scenario())
    assert result["c0"] == [0.0, 0.0]
    assert result["c1"] == [1.0, 1.0]
    assert result["c2"] == [2.0, 2.0]


def test_length_mismatch_raises_instead_of_padding_or_faking_a_vector():
    comments = [_comment(i) for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0]}]})

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(comments, client=client, api_key=API_KEY)

    with pytest.raises(stage_a.EmbeddingArrayLengthMismatchError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.expected == 3
    assert exc_info.value.got == 1


def test_non_200_response_raises_api_error_not_a_silent_fallback():
    comments = [_comment(0)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid api key")

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(comments, client=client, api_key=API_KEY)

    with pytest.raises(stage_a.EmbeddingAPIError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 401


def test_embed_all_comments_batches_at_the_configured_size():
    comments = [_comment(i) for i in range(10)]
    seen_batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        n = len(body["input"])
        seen_batch_sizes.append(n)
        return httpx.Response(200, json={
            "data": [{"index": i, "embedding": [float(i)]} for i in range(n)]
        })

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_all_comments(
                comments, client=client, api_key=API_KEY, batch_size=4
            )

    result = asyncio.run(scenario())
    assert set(result.keys()) == {c.id for c in comments}
    assert seen_batch_sizes == [4, 4, 2]


def test_request_uses_bearer_auth_header_not_a_query_param():
    comments = [_comment(0)]
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    async def scenario():
        async with _client(handler) as client:
            await stage_a.embed_comments_batch(comments, client=client, api_key=API_KEY)

    asyncio.run(scenario())
    assert captured["auth"] == f"Bearer {API_KEY}"
