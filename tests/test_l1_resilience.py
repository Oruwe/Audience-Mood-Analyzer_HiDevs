"""L1 unit tests — resilience.retry_transient (SPEC §8 retry policy).

Uses tiny explicit wait bounds so these stay fast regardless of the
production defaults used elsewhere (ingestion/youtube.py, engine/batching.py,
engine/stage_a.py) — this tests the retry *mechanism* in isolation, not
those call sites' real backoff timing.
"""

import asyncio

import httpx
import pytest
from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    UnprocessableEntityError,
)

from resilience import (
    is_fallback_worthy_api_error,
    is_retryable_api_error,
    retry_transient,
    retry_transient_api_error,
)


class _Boom(Exception):
    pass


class _OtherError(Exception):
    pass


def _fast(*exception_types, max_attempts=4):
    return retry_transient(
        *exception_types, max_attempts=max_attempts, wait_multiplier=0.001, wait_max=0.01
    )


def test_retries_until_success_within_the_attempt_budget():
    calls = {"count": 0}

    @_fast(_Boom, max_attempts=4)
    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise _Boom("not yet")
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert calls["count"] == 3


def test_gives_up_and_reraises_the_original_exception_after_max_attempts():
    calls = {"count": 0}

    @_fast(_Boom, max_attempts=3)
    async def always_fails():
        calls["count"] += 1
        raise _Boom("nope")

    with pytest.raises(_Boom):
        asyncio.run(always_fails())
    assert calls["count"] == 3


def test_non_retryable_exception_types_propagate_immediately():
    calls = {"count": 0}

    @_fast(_Boom, max_attempts=4)
    async def wrong_error():
        calls["count"] += 1
        raise _OtherError("not the retryable type")

    with pytest.raises(_OtherError):
        asyncio.run(wrong_error())
    assert calls["count"] == 1  # never retried


def test_works_on_sync_functions_too():
    calls = {"count": 0}

    @_fast(_Boom, max_attempts=3)
    def flaky_sync():
        calls["count"] += 1
        if calls["count"] < 2:
            raise _Boom("not yet")
        return "ok"

    assert flaky_sync() == "ok"
    assert calls["count"] == 2


# ---------------------------------------------------------------------------
# is_retryable_api_error / retry_transient_api_error -- a deny-list, not an
# allow-list, after a live incident where a real transient OpenRouter
# failure ("Nvidia: Service temporarily overloaded") surfaced from litellm
# as the bare base APIError rather than one of the more specific
# subclasses this project used to retry on by name.
# ---------------------------------------------------------------------------

def _api_error(cls=APIError):
    # Every litellm exception subclass takes message/llm_provider/model;
    # the bare APIError additionally requires status_code, while
    # PermissionDeniedError/UnprocessableEntityError require a real
    # `response` (no default) instead.
    kwargs = dict(message="boom", llm_provider="openrouter", model="m")
    if cls is APIError:
        kwargs["status_code"] = 500
    elif cls in (PermissionDeniedError, UnprocessableEntityError):
        kwargs["response"] = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    return cls(**kwargs)


@pytest.mark.parametrize("cls", [APIError, ServiceUnavailableError])
def test_is_retryable_api_error_true_for_transient_types(cls):
    assert is_retryable_api_error(_api_error(cls)) is True


@pytest.mark.parametrize("cls", [AuthenticationError, BadRequestError, NotFoundError])
def test_is_retryable_api_error_false_for_permanent_client_errors(cls):
    assert is_retryable_api_error(_api_error(cls)) is False


def test_is_retryable_api_error_false_for_rate_limit_error():
    """Latency fix, not correctness (measured live, 2026-09-13): a
    free-tier 429 is a *sustained* shared-pool exhaustion, not a momentary
    blip -- retrying the SAME model wastes ~20-30s before giving up on it
    anyway. is_fallback_worthy_api_error (below) still treats it as
    fallback-worthy, so the net effect is "skip straight to the fallback
    model" instead of "retry, then fall back"."""
    assert is_retryable_api_error(_api_error(RateLimitError)) is False


def test_is_retryable_api_error_false_for_a_non_api_error():
    assert is_retryable_api_error(ValueError("not even an APIError")) is False


def test_retry_transient_api_error_retries_a_bare_api_error():
    calls = {"count": 0}

    @retry_transient_api_error(max_attempts=4, wait_multiplier=0.001, wait_max=0.01)
    async def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise _api_error()
        return "ok"

    assert asyncio.run(flaky()) == "ok"
    assert calls["count"] == 3


def test_is_retryable_api_error_false_for_not_found_error():
    """Live incident (2026-09-13): a 404 means retrying the SAME model is
    pointless (it's gone/withdrawn), but a *different* model may still
    work -- see is_fallback_worthy_api_error below, which disagrees with
    this one on purpose."""
    assert is_retryable_api_error(_api_error(NotFoundError)) is False


@pytest.mark.parametrize("cls", [APIError, RateLimitError, ServiceUnavailableError, NotFoundError])
def test_is_fallback_worthy_api_error_true_including_not_found(cls):
    assert is_fallback_worthy_api_error(_api_error(cls)) is True


@pytest.mark.parametrize("cls", [AuthenticationError, BadRequestError, PermissionDeniedError, UnprocessableEntityError])
def test_is_fallback_worthy_api_error_false_for_request_or_account_level_errors(cls):
    """These are about the caller's own request/account (bad key,
    malformed body, permission, content policy) and would fail identically
    against any model -- falling back to a different one doesn't help."""
    assert is_fallback_worthy_api_error(_api_error(cls)) is False


def test_is_fallback_worthy_api_error_false_for_a_non_api_error():
    assert is_fallback_worthy_api_error(ValueError("not even an APIError")) is False


def test_retry_transient_api_error_does_not_retry_authentication_error():
    calls = {"count": 0}

    @retry_transient_api_error(max_attempts=4, wait_multiplier=0.001, wait_max=0.01)
    async def always_bad_key():
        calls["count"] += 1
        raise _api_error(AuthenticationError)

    with pytest.raises(AuthenticationError):
        asyncio.run(always_bad_key())
    assert calls["count"] == 1
