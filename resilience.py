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

import openai
import tenacity
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)

DEFAULT_MAX_ATTEMPTS = 4        # 1 try + up to 3 retries
DEFAULT_WAIT_MULTIPLIER = 0.5   # seconds
DEFAULT_WAIT_MAX = 8.0          # seconds

# Live incident (2026-09-13): a real "Nvidia: Service temporarily overloaded"
# upstream failure surfaced from litellm as its own catch-all
# `litellm.exceptions.APIError`, not one of the more specific subclasses
# (RateLimitError, ServiceUnavailableError, ...) engine/batching.py and
# engine/insights.py used to retry on by name -- OpenRouter doesn't always
# wrap a transient upstream failure into a specific subclass, so that
# allow-list under-retried.
#
# The check below is deliberately against `openai.APIError`, not litellm's
# own `litellm.exceptions.APIError` -- confirmed by inspecting the actual
# MRO: every litellm exception (its own APIError included) inherits from
# `openai.APIError`/`openai.OpenAIError`, but litellm's `APIError` is a
# *sibling* of RateLimitError/ServiceUnavailableError/etc., not their
# parent, so checking against it would miss most of them entirely.
#
# NON_RETRYABLE_API_ERRORS is the opposite shape from an allow-list: every
# `openai.APIError` is retryable EXCEPT these few, which mean a genuine,
# permanent client mistake (bad key, malformed request, permission,
# not-found, content-policy -- BadRequestError's own subclass) that
# retrying or falling back to a different model would never fix.
NON_RETRYABLE_API_ERRORS = (
    AuthenticationError, BadRequestError, NotFoundError,
    PermissionDeniedError, UnprocessableEntityError,
)


def is_retryable_api_error(exc: BaseException) -> bool:
    """True for an OpenAI/litellm API error worth retrying or falling back
    to a different model on -- see NON_RETRYABLE_API_ERRORS above for why
    this is a deny-list, not an allow-list of specific subclasses."""
    return isinstance(exc, openai.APIError) and not isinstance(exc, NON_RETRYABLE_API_ERRORS)


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


def retry_transient_api_error(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    wait_multiplier: float = DEFAULT_WAIT_MULTIPLIER,
    wait_max: float = DEFAULT_WAIT_MAX,
):
    """Like `retry_transient`, but for litellm's OpenRouter call sites
    specifically: retries any `is_retryable_api_error(exc)` rather than a
    fixed list of subclasses passed in by name. See that function's
    docstring for why an allow-list under-retries here.
    """
    return tenacity.retry(
        retry=tenacity.retry_if_exception(is_retryable_api_error),
        wait=tenacity.wait_random_exponential(multiplier=wait_multiplier, max=wait_max),
        stop=tenacity.stop_after_attempt(max_attempts),
        reraise=True,
    )
