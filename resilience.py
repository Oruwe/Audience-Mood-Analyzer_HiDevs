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
    RateLimitError,
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
# content-policy -- BadRequestError's own subclass) that retrying the SAME
# model would never fix.
#
# Live incident (2026-09-13, on top of the one above): a Stage B call
# 404'd with "This model is unavailable for free ... use this slug
# instead: minimax/minimax-m2.7" -- OpenRouter had withdrawn the specific
# `:free` slug entirely, not rate-limited it. NotFoundError was originally
# in this deny-list and shared by both retry and fallback decisions, which
# meant the (correct) "don't retry the same now-nonexistent model" call
# also (incorrectly) blocked ever falling back to a *different* model --
# exactly the situation config.models' fallback constants exist for.
# NotFoundError is therefore excluded from retry (below) but deliberately
# NOT excluded from fallback (NON_FALLBACK_API_ERRORS, further down): the
# other four types are about *our own request or account* (bad key,
# malformed body, permission, content policy) and would fail identically
# against any model, so falling back doesn't help those -- but "this model
# doesn't exist/isn't available" is specific to the one model string, and
# is exactly what trying a different one fixes.
#
# RateLimitError joined this list for a latency reason, not a correctness
# one (measured live, 2026-09-13): a free-tier 429's own error body says
# "temporarily rate-limited upstream ... upstream_provider_shared_pool" --
# a *sustained* shared-capacity exhaustion, confirmed by watching a real
# job hit the same 429 on the same model dozens of times over several
# minutes. Spending up to DEFAULT_MAX_ATTEMPTS retries with exponential
# backoff (~20-30s worst case) against a condition that won't clear in
# that window is pure wasted latency; falling back to a different model
# immediately (still covered by NON_FALLBACK_API_ERRORS not listing it)
# is strictly faster and no less correct.
NON_RETRYABLE_API_ERRORS = (
    AuthenticationError, BadRequestError, NotFoundError,
    PermissionDeniedError, RateLimitError, UnprocessableEntityError,
)

NON_FALLBACK_API_ERRORS = (
    AuthenticationError, BadRequestError, PermissionDeniedError, UnprocessableEntityError,
)


def is_retryable_api_error(exc: BaseException) -> bool:
    """True for an OpenAI/litellm API error worth retrying the SAME model
    on -- see NON_RETRYABLE_API_ERRORS above for why this is a deny-list,
    not an allow-list of specific subclasses."""
    return isinstance(exc, openai.APIError) and not isinstance(exc, NON_RETRYABLE_API_ERRORS)


def is_fallback_worthy_api_error(exc: BaseException) -> bool:
    """True for an OpenAI/litellm API error worth falling back to a
    *different* model on. Deliberately more permissive than
    is_retryable_api_error: a NotFoundError ("this model/slug doesn't
    exist or isn't available") is exactly what a different model fixes,
    even though retrying the same one again obviously wouldn't."""
    return isinstance(exc, openai.APIError) and not isinstance(exc, NON_FALLBACK_API_ERRORS)


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
