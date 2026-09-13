"""Shared SPEC §4.1b batching guard.

"[Required guard:] validate that the returned array length equals the
input batch length. Batched structured output silently drops or merges
items under load. On mismatch, split the batch in half and retry."

Originally specified with Stage B in mind. The §4.1 amendment moving
Stage A onto OpenRouter too (config/models.py) put Stage A under the exact
same failure mode, so engine/stage_a.py (Stage A sentiment) and
engine/llm_client.py (Stage B classification) share this implementation
rather than each re-solving the recursive split-and-retry logic.

Every stage's response schema must follow one shape:
    class SomeBatch(BaseModel):
        results: list[SomeItem]   # each SomeItem has a `comment_id: str`
"""

from __future__ import annotations

import logging

from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel, ValidationError

from resilience import retry_transient
from schemas import RawComment

logger = logging.getLogger(__name__)


class BatchClassificationFailedError(RuntimeError):
    """A single-item batch still failed the guard after splitting as far as
    it can go. Stage modules should subclass this for a stage-specific
    error type while sharing this module's raising logic.
    """


def _parse(raw_content: str, response_schema: type[BaseModel]) -> BaseModel | None:
    try:
        return response_schema.model_validate_json(raw_content)
    except (ValidationError, ValueError):
        return None


# SPEC §8: "Retries: tenacity, exponential backoff + jitter, on ... OpenRouter
# calls." Only these -- a genuine bad request, auth failure, or content-policy
# rejection must fail immediately, not retry into a slower failure.
@retry_transient(
    RateLimitError, ServiceUnavailableError, Timeout,
    APIConnectionError, InternalServerError, BadGatewayError,
)
async def _call_model(model: str, api_key: str, messages: list[dict], response_schema: type[BaseModel]):
    return await acompletion(
        model=model,
        api_key=api_key,
        messages=messages,
        response_format=response_schema,
        timeout=30,
    )


async def _call_model_with_fallback(
    models: tuple[str, ...], api_key: str, messages: list[dict],
    response_schema: type[BaseModel], *, stage_label: str,
):
    """Try *models* in order, each already retried on its own transient
    errors by `_call_model` -- this only steps to the next model once a
    given one has exhausted those retries. Added after a live incident
    (2026-09-13, config/models.py's fallback constants): a free OpenRouter
    model's 429 can be a *sustained* shared-pool exhaustion, not a
    momentary blip, and outlasts resilience.py's ~8s retry window. A
    second free model from a different vendor is unlikely to be exhausted
    by the same event.
    """
    last_exc: Exception | None = None
    for i, model in enumerate(models):
        try:
            return await _call_model(model, api_key, messages, response_schema)
        except (RateLimitError, ServiceUnavailableError) as exc:
            last_exc = exc
            if i + 1 < len(models):
                logger.warning(
                    "%s: %s unavailable after retries (%s); falling back to %s",
                    stage_label, model, exc, models[i + 1],
                )
    assert last_exc is not None  # unreachable with a non-empty models tuple
    raise last_exc


async def run_batched_llm_classification(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    response_schema: type[BaseModel],
    stage_label: str,
    error_cls: type[BatchClassificationFailedError] = BatchClassificationFailedError,
    fallback_models: tuple[str, ...] = (),
) -> dict[str, BaseModel]:
    """Classify one batch, applying the §4.1b split-and-retry guard.

    Returns a dict keyed by comment_id, covering exactly the input
    comments — guaranteed by the guard below, never a partial or
    misaligned result. *response_schema* must have a `results: list[Item]`
    field, each `Item` carrying a `comment_id: str`.

    *fallback_models*, if given, are tried in order after *model* has
    exhausted its own retries on a RateLimitError/ServiceUnavailableError
    (see `_call_model_with_fallback`) — a free OpenRouter model's shared
    capacity pool can stay exhausted well past resilience.py's retry
    window; a different vendor's free model is unlikely to be exhausted by
    the same event.
    """
    if not comments:
        return {}

    lines = "\n".join(f"{c.id}: {c.text}" for c in comments)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": lines},
    ]
    response = await _call_model_with_fallback(
        (model, *fallback_models), api_key, messages, response_schema, stage_label=stage_label,
    )
    parsed = _parse(response.choices[0].message.content, response_schema)

    expected_ids = {c.id for c in comments}
    if parsed is not None and len(parsed.results) == len(comments):
        got_ids = {item.comment_id for item in parsed.results}
        if got_ids == expected_ids:
            return {item.comment_id: item for item in parsed.results}

    # SPEC §4.1b: "On mismatch, split the batch in half and retry."
    if len(comments) == 1:
        raise error_cls(
            f"{stage_label} classification failed for comment "
            f"{comments[0].id!r} even as a single-item batch."
        )
    logger.warning(
        "%s batch of %d returned a length/id mismatch; splitting and retrying",
        stage_label, len(comments),
    )
    mid = len(comments) // 2
    left = await run_batched_llm_classification(
        comments[:mid], api_key=api_key, model=model, system_prompt=system_prompt,
        response_schema=response_schema, stage_label=stage_label, error_cls=error_cls,
        fallback_models=fallback_models,
    )
    right = await run_batched_llm_classification(
        comments[mid:], api_key=api_key, model=model, system_prompt=system_prompt,
        response_schema=response_schema, stage_label=stage_label, error_cls=error_cls,
        fallback_models=fallback_models,
    )
    return {**left, **right}


async def classify_all_batches(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    response_schema: type[BaseModel],
    stage_label: str,
    batch_size: int,
    error_cls: type[BatchClassificationFailedError] = BatchClassificationFailedError,
    fallback_models: tuple[str, ...] = (),
) -> dict[str, BaseModel]:
    """Classify every comment in *comments*, chunked at *batch_size* per call."""
    results: dict[str, BaseModel] = {}
    for start in range(0, len(comments), batch_size):
        batch = comments[start : start + batch_size]
        results.update(await run_batched_llm_classification(
            batch, api_key=api_key, model=model, system_prompt=system_prompt,
            response_schema=response_schema, stage_label=stage_label, error_cls=error_cls,
            fallback_models=fallback_models,
        ))
    return results
