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
from pydantic import BaseModel, ValidationError

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


async def run_batched_llm_classification(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    response_schema: type[BaseModel],
    stage_label: str,
    error_cls: type[BatchClassificationFailedError] = BatchClassificationFailedError,
) -> dict[str, BaseModel]:
    """Classify one batch, applying the §4.1b split-and-retry guard.

    Returns a dict keyed by comment_id, covering exactly the input
    comments — guaranteed by the guard below, never a partial or
    misaligned result. *response_schema* must have a `results: list[Item]`
    field, each `Item` carrying a `comment_id: str`.
    """
    if not comments:
        return {}

    lines = "\n".join(f"{c.id}: {c.text}" for c in comments)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": lines},
    ]
    response = await acompletion(
        model=model,
        api_key=api_key,
        messages=messages,
        response_format=response_schema,
        timeout=30,
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
    )
    right = await run_batched_llm_classification(
        comments[mid:], api_key=api_key, model=model, system_prompt=system_prompt,
        response_schema=response_schema, stage_label=stage_label, error_cls=error_cls,
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
) -> dict[str, BaseModel]:
    """Classify every comment in *comments*, chunked at *batch_size* per call."""
    results: dict[str, BaseModel] = {}
    for start in range(0, len(comments), batch_size):
        batch = comments[start : start + batch_size]
        results.update(await run_batched_llm_classification(
            batch, api_key=api_key, model=model, system_prompt=system_prompt,
            response_schema=response_schema, stage_label=stage_label, error_cls=error_cls,
        ))
    return results
