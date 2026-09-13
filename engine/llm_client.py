"""SPEC §4.1 Stage B: batched, generative classification of the
Stage-A-flagged subset (~10-20% of comments) — intent / is_request /
is_confusion, 40-60 comments per call.

Kept from V2 (SPEC §2 "keep the routing and the contract enforcement"):
the tiered-failover-with-Pydantic-validation *shape* this module has
always had. Rewritten (SPEC §2 "rewrite only the call shape", §14 step 5):
the call now targets a single OpenRouter model
(config.models.STAGE_B_CLASSIFY) via litellm's openrouter/* routing,
returning Stage B's own intent/is_request/is_confusion schema, not V2's
Gemini->Groq DeepMoodAnalysis chain (brand-monitoring fields like
bug_report/pricing_complaint/escalate_to_pr don't fit a creator product,
and that whole call shape was one-comment-per-call, exactly what SPEC §4.1
says not to do). No multi-provider failover chain here: SPEC §7 locks all
generative inference to OpenRouter, whose own platform-level routing is now
the resilience layer; app-level retries (tenacity, exponential backoff)
are SPEC §8's job, a later phase.

Applies the SPEC §4.1b guard via engine.batching (shared with Stage A,
engine/stage_a.py, since both are the same "batched map, N-in-N-out"
shape): exact array length AND comment-id set must match the input batch,
or it splits in half and retries.
"""

from __future__ import annotations

from config.models import STAGE_B_CLASSIFY
from engine.batching import BatchClassificationFailedError, classify_all_batches
from schemas import RawComment, StageBClassificationBatch, StageBClassificationItem

DEFAULT_BATCH_SIZE = 50  # SPEC §4.1: "40-60 comments per call"

SYSTEM_PROMPT = (
    "You are analysing YouTube comments for a content creator. For every "
    "comment given, decide: its primary intent, whether it is a request for "
    "future content, and whether it expresses confusion about something in "
    "the video. Respond ONLY with JSON matching exactly this schema: "
    '{"results": [{"comment_id": "<id>", '
    '"intent": "request|confusion|praise|criticism|other", '
    '"is_request": <true|false>, "is_confusion": <true|false>}, ...]}. '
    "Return exactly one result object per input comment, each carrying that "
    "input's exact comment_id — never add, drop, merge, or reorder them."
)


class StageBError(RuntimeError):
    """Base class for Stage B errors raised on purpose."""


class ClassificationBatchFailedError(StageBError, BatchClassificationFailedError):
    """A single-comment batch still failed after splitting as far as it can."""


async def classify_batch(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str = STAGE_B_CLASSIFY,
) -> dict[str, StageBClassificationItem]:
    """Classify one batch, applying the SPEC §4.1b split-and-retry guard
    (engine.batching — shared with Stage A).

    Returns a dict keyed by comment_id, covering exactly the input comments
    — guaranteed by the guard, never a partial or misaligned result.
    """
    return await classify_all_batches(
        comments,
        api_key=api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_schema=StageBClassificationBatch,
        stage_label="Stage B",
        batch_size=len(comments) or 1,  # one call for this whole batch, no chunking
        error_cls=ClassificationBatchFailedError,
    )


async def classify_all(
    comments: list[RawComment],
    *,
    api_key: str,
    model: str = STAGE_B_CLASSIFY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, StageBClassificationItem]:
    """Classify every comment in *comments*, chunked at *batch_size* per call.

    Callers should pass only the Stage-A-flagged subset (SPEC §4.1: "only
    what Stage A flags") — this function itself does no filtering; see
    engine/stage_filter.py for that.
    """
    return await classify_all_batches(
        comments,
        api_key=api_key,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        response_schema=StageBClassificationBatch,
        stage_label="Stage B",
        batch_size=batch_size,
        error_cls=ClassificationBatchFailedError,
    )
