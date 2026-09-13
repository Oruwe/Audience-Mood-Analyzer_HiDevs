"""L1 unit tests — resilience.retry_transient (SPEC §8 retry policy).

Uses tiny explicit wait bounds so these stay fast regardless of the
production defaults used elsewhere (ingestion/youtube.py, engine/batching.py,
engine/stage_a.py) — this tests the retry *mechanism* in isolation, not
those call sites' real backoff timing.
"""

import asyncio

import pytest

from resilience import retry_transient


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
