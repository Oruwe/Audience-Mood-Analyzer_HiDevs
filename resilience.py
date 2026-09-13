"""SPEC §8: "Retries: tenacity, exponential backoff + jitter, on YouTube
and OpenRouter calls."

One shared decorator factory so ingestion/youtube.py and the OpenRouter
call sites (engine/batching.py, engine/stage_a.py's embeddings call) don't
each reinvent the policy. Deliberately narrow: only the exception types the
caller names are retried — a genuine data problem (a 400, a schema
mismatch, a quota refusal) must never be silently retried into a slower
failure; only transient network/provider trouble should be.

Default wait bounds are modest (not minutes-long backoff) on purpose: this
runs inside a background job a user is watching an `st.status` bar for
(SPEC §8/§5), not a fire-and-forget nightly batch — a slow-but-bounded
retry is fine, an unbounded one reads as a hang.
"""

from __future__ import annotations

import tenacity

DEFAULT_MAX_ATTEMPTS = 4        # 1 try + up to 3 retries
DEFAULT_WAIT_MULTIPLIER = 0.5   # seconds
DEFAULT_WAIT_MAX = 8.0          # seconds


def retry_transient(
    *exception_types: type[BaseException],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER,
    wait_max: float = DEFAULT_WAIT_MAX,
):
    """Retry only on *exception_types*, exponential backoff + jitter,
    re-raising the original exception once attempts are exhausted. Works
    as a decorator on both sync and async functions.
    """
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(exception_types),
        wait=tenacity.wait_random_exponential(multiplier=wait_multiplier, max=wait_max),
        stop=tenacity.stop_after_attempt(max_attempts),
        reraise=True,
    )
