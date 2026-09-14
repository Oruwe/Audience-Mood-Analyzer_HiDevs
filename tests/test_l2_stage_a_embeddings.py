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


def test_the_openrouter_routing_prefix_is_stripped_before_the_real_request():
    """Shipped as a real bug once already: config.models' STAGE_A_EMBEDDINGS
    carries litellm's `openrouter/` routing prefix (needed by the sentiment
    call, which goes through litellm), but this function talks to
    OpenRouter's REST API directly -- sending that prefix verbatim got a
    live "Model openrouter/vendor/... does not exist" from the real API.
    Uses a made-up model string (not the real configured one) so this test
    doesn't itself trip the single-config-boundary rule.
    """
    comments = [_comment(0)]
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    async def scenario():
        async with _client(handler) as client:
            await stage_a.embed_comments_batch(
                comments, client=client, api_key=API_KEY,
                model="openrouter/some-vendor/some-embedding-model",
            )

    asyncio.run(scenario())
    assert captured["model"] == "some-vendor/some-embedding-model"


def test_a_model_id_without_the_prefix_is_left_untouched():
    comments = [_comment(0)]
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        captured["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    async def scenario():
        async with _client(handler) as client:
            await stage_a.embed_comments_batch(
                comments, client=client, api_key=API_KEY,
                model="some-vendor/some-embedding-model",
            )

    asyncio.run(scenario())
    assert captured["model"] == "some-vendor/some-embedding-model"


# ---------------------------------------------------------------------------
# Fallback-on-failure. Live incident (2026-09-13): the configured embedding
# model 404'd with OpenRouter's own "No endpoints found" -- and, before
# this, embeddings had NO fallback mechanism at all, so that 404 would have
# killed an entire real analysis outright (caught for $0.00015 by
# harness/preflight.py instead). These prove the fix: a fallback-worthy
# failure on the primary tries the next model; a request-level failure
# (bad key) does not, because it would fail identically on any model.
# ---------------------------------------------------------------------------

def test_falls_back_to_a_different_model_on_a_404():
    comments = [_comment(0)]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = __import__("json").loads(request.content)["model"]
        calls.append(model)
        if model == "model-a":
            return httpx.Response(404, json={"error": {"message": "No endpoints found for model-a"}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(
                comments, client=client, api_key=API_KEY,
                model="openrouter/model-a", fallback_models=("openrouter/model-b",),
            )

    result = asyncio.run(scenario())

    assert calls == ["model-a", "model-b"]
    assert result == {"c0": [0.1, 0.2]}


def test_does_not_fall_back_on_a_401_bad_key():
    """A bad API key fails identically against any model -- falling back
    wastes a call and still fails, so it must not be tried."""
    comments = [_comment(0)]
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(__import__("json").loads(request.content)["model"])
        return httpx.Response(401, text="invalid api key")

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(
                comments, client=client, api_key=API_KEY,
                model="openrouter/model-a", fallback_models=("openrouter/model-b",),
            )

    with pytest.raises(stage_a.EmbeddingAPIError) as exc_info:
        asyncio.run(scenario())

    assert exc_info.value.status_code == 401
    assert calls == ["model-a"]  # never tried model-b


def test_raises_the_last_error_when_every_model_including_fallbacks_fails():
    comments = [_comment(0)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "no endpoints"}})

    async def scenario():
        async with _client(handler) as client:
            return await stage_a.embed_comments_batch(
                comments, client=client, api_key=API_KEY,
                model="openrouter/model-a", fallback_models=("openrouter/model-b",),
            )

    with pytest.raises(stage_a.EmbeddingAPIError) as exc_info:
        asyncio.run(scenario())
    assert exc_info.value.status_code == 404


def test_default_fallback_is_the_configured_stage_a_embeddings_fallback():
    """No caller in production passes fallback_models explicitly (same
    pattern as the other three stages) -- the default must be the real
    configured constant, not an empty tuple that silently disables the
    protection this was built for."""
    import inspect
    default = inspect.signature(stage_a.embed_comments_batch).parameters["fallback_models"].default
    assert default == (stage_a.STAGE_A_EMBEDDINGS_FALLBACK,)
