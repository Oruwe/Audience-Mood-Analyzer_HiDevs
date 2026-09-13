"""SPEC §4.1 Stage A: sentiment classification + embeddings over 100% of
comments.

Architecture deviation (2026-09-13): SPEC §4.1 originally specified Stage A
as a local, free, deterministic encoder. This build's operator explicitly
chose to route it through OpenRouter instead, after being told this drops
determinism, the "runs on 100% of comments for free" cost model, and the
§10 prompt-injection-immunity argument that only holds for a pure encoder
with no instruction channel — see config/models.py's module docstring and
SPEC.md §4.1/§10 for the full accepted-tradeoff record.

This module still does the two jobs SPEC §4.1 assigns to Stage A:

1. classify_sentiment_batch / classify_all_sentiments — one sentiment label
   per comment, via a batched chat completion (litellm, an `openrouter/*`
   model from config.models).
2. embed_comments_batch / embed_all_comments — one embedding vector per
   comment, via OpenRouter's OpenAI-compatible `/embeddings` endpoint,
   called directly over httpx rather than through litellm: litellm's
   maintained cost map (used to pick every other model in this project)
   has zero `openrouter/*` embedding-mode entries as of this build, so
   routing an embedding call through litellm for this provider is
   unverified.

Both batch functions apply the SPEC §4.1b guard — originally written with
Stage B in mind, but it applies just as much here now that Stage A is also
a batched generative call: the response must return exactly one result per
input comment, matched by `comment_id`, never by list position. Sentiment
batches retry via split-in-half on any mismatch, per §4.1b's exact wording
("On mismatch, split the batch in half and retry"). The embeddings
endpoint isn't LLM-generated free-form JSON, so a count mismatch there is a
hard error instead — and it is a hard error, deliberately: SPEC §4.1 calls
out V2's old hash-vector fallback by name as the mistake not to repeat,
because it "silently made clustering meaningless whenever it fired."
"""

from __future__ import annotations

import logging

import httpx
from litellm import acompletion
from pydantic import ValidationError

from config.models import STAGE_A_EMBEDDINGS, STAGE_A_SENTIMENT
from schemas import RawComment, StageASentimentBatch, StageASentimentItem

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

DEFAULT_SENTIMENT_BATCH_SIZE = 50   # SPEC §4.1's 40-60/call range for the cascade
DEFAULT_EMBEDDING_BATCH_SIZE = 100  # embeddings are cheap/small; batch generously

SENTIMENT_SYSTEM_PROMPT = (
    "You are a sentiment classifier for YouTube comments. For every comment "
    "given, return exactly one result. Respond ONLY with JSON matching "
    'exactly this schema: {"results": [{"comment_id": "<id>", '
    '"sentiment": "strongly_positive|positive|neutral|negative|'
    'critical_escalation", "confidence": <0.0-1.0>}, ...]}. Return exactly '
    "one result object per input comment, each carrying that input's exact "
    "comment_id — never add, drop, merge, or reorder them."
)


class StageAError(RuntimeError):
    """Base class for Stage A errors raised on purpose."""


class SentimentBatchFailedError(StageAError):
    """A single-comment batch still failed after splitting as far as it can."""


class EmbeddingArrayLengthMismatchError(StageAError):
    """The embeddings endpoint returned a different count than requested."""

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"Requested {expected} embeddings, got {got} back.")


class EmbeddingAPIError(StageAError):
    """The embeddings endpoint returned a non-2xx response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"OpenRouter embeddings -> HTTP {status_code}: {body[:500]}")


def _build_sentiment_messages(comments: list[RawComment]) -> list[dict]:
    lines = "\n".join(f"{c.id}: {c.text}" for c in comments)
    return [
        {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
        {"role": "user", "content": lines},
    ]


def _parse_sentiment_response(raw_content: str) -> StageASentimentBatch | None:
    try:
        return StageASentimentBatch.model_validate_json(raw_content)
    except (ValidationError, ValueError):
        return None


async def classify_sentiment_batch(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str = STAGE_A_SENTIMENT,
) -> dict[str, StageASentimentItem]:
    """Classify one batch, applying the SPEC §4.1b split-and-retry guard.

    Returns a dict keyed by comment_id, covering exactly the input comments
    — guaranteed by the guard below, never a partial or misaligned result.
    """
    if not comments:
        return {}

    response = await acompletion(
        model=model,
        api_key=api_key,
        messages=_build_sentiment_messages(comments),
        response_format=StageASentimentBatch,
        timeout=30,
    )
    parsed = _parse_sentiment_response(response.choices[0].message.content)

    expected_ids = {c.id for c in comments}
    if parsed is not None and len(parsed.results) == len(comments):
        got_ids = {item.comment_id for item in parsed.results}
        if got_ids == expected_ids:
            return {item.comment_id: item for item in parsed.results}

    # SPEC §4.1b: "On mismatch, split the batch in half and retry."
    if len(comments) == 1:
        raise SentimentBatchFailedError(
            f"Sentiment classification failed for comment {comments[0].id!r} "
            "even as a single-item batch."
        )
    logger.warning(
        "Sentiment batch of %d returned a length/id mismatch; splitting and retrying",
        len(comments),
    )
    mid = len(comments) // 2
    left = await classify_sentiment_batch(comments[:mid], api_key=api_key, model=model)
    right = await classify_sentiment_batch(comments[mid:], api_key=api_key, model=model)
    return {**left, **right}


async def classify_all_sentiments(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str = STAGE_A_SENTIMENT,
    batch_size: int = DEFAULT_SENTIMENT_BATCH_SIZE,
) -> dict[str, StageASentimentItem]:
    """Classify every comment, batched at *batch_size* per call."""
    results: dict[str, StageASentimentItem] = {}
    for start in range(0, len(comments), batch_size):
        batch = comments[start : start + batch_size]
        results.update(await classify_sentiment_batch(batch, api_key=api_key, model=model))
    return results


async def embed_comments_batch(
    comments: list[RawComment],
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str = STAGE_A_EMBEDDINGS,
) -> dict[str, list[float]]:
    """One embedding vector per comment, via OpenRouter's `/embeddings`
    endpoint. Raises rather than silently degrading on any mismatch or
    error — see the module docstring on why that's deliberate.
    """
    if not comments:
        return {}

    response = await client.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "input": [c.text for c in comments]},
    )
    if response.status_code != 200:
        raise EmbeddingAPIError(response.status_code, response.text)

    items = response.json().get("data", [])
    if len(items) != len(comments):
        raise EmbeddingArrayLengthMismatchError(expected=len(comments), got=len(items))

    # OpenAI-compatible response: each item carries its input "index", not
    # necessarily returned in request order — sort rather than assume.
    items_by_index = sorted(items, key=lambda item: item.get("index", 0))
    return {
        comment.id: item["embedding"]
        for comment, item in zip(comments, items_by_index, strict=True)
    }


async def embed_all_comments(
    comments: list[RawComment],
    *,
    client: httpx.AsyncClient,
    api_key: str,
    model: str = STAGE_A_EMBEDDINGS,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> dict[str, list[float]]:
    """Embed every comment, batched at *batch_size* per call."""
    vectors: dict[str, list[float]] = {}
    for start in range(0, len(comments), batch_size):
        batch = comments[start : start + batch_size]
        vectors.update(
            await embed_comments_batch(batch, client=client, api_key=api_key, model=model)
        )
    return vectors
