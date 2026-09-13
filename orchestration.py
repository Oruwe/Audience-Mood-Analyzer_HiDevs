"""SPEC §8 — Orchestration.

`analyze_channel(job_id, channel_ref)` composes the full pipeline —
ingestion -> Stage A (sentiment + embeddings) -> the Stage A/B filter ->
Stage B classification -> Stage C synthesis (engine/insights.py) — as one
checkpointed, cancellable, resumable job. Stage C is checkpointed as a
single unit (not per-cluster like Stage A/B's chunks): it's only ~8 calls,
cheap enough that re-running the whole thing on resume isn't worth the
extra bookkeeping a finer-grained scheme would need.

Replaces pipeline.py (SPEC §2 cut: "3-process firehose runner + token
bucket — replaced by a request-scoped job"). This is that job.

Checkpointing (SPEC §8, required): each unit of work — one video's
comments, one Stage A sentiment batch, one Stage A embedding batch, one
Stage B batch — is recorded to Postgres (storage.postgres) as it finishes.
On (re)start, the job loads what's already checkpointed and skips it, so a
crash or a Streamlit Cloud sleep-then-wake never re-spends YouTube/
OpenRouter quota on work already done. `total_units`/`completed_units` are
scoped to the *current* stage (see `stage` alongside them), not a single
monotonic count across four differently-shaped stages.

Background execution: `run_job_in_background` starts a daemon thread
running its own `asyncio.run(...)` event loop — asyncpg connections are
bound to the loop that created them (storage/postgres.py), so each job
needs its own fresh one rather than sharing one with the Streamlit main
thread.

Cancellation: cooperative. The loop checks `is_cancel_requested` between
units of work and raises JobCancelledError to unwind cleanly — nothing is
force-killed mid-write, so a cancelled job's checkpoints stay consistent.

Atomic claim: `claim_job` (storage/postgres.py) transitions a job from
'pending' to 'running' exactly once. `run_job_in_background`/
`_run_claimed_job` always claims before doing any work, so calling it
twice for the same job_id — a double-click, a re-triggered background
thread after a Streamlit rerun — is a safe no-op the second time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

import httpx

from engine.insights import build_channel_insights
from engine.llm_client import DEFAULT_BATCH_SIZE as STAGE_B_BATCH_SIZE
from engine.llm_client import classify_batch as stage_b_classify_batch
from engine.stage_a import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_SENTIMENT_BATCH_SIZE,
    classify_sentiment_batch,
    embed_comments_batch,
)
from engine.stage_filter import select_for_stage_b
from ingestion.dedup import ContentDeduplicator
from ingestion.youtube import (
    DEFAULT_MAX_VIDEOS,
    QuotaLedger,
    estimate_channel_analysis,
    fetch_video_comments,
)
from schemas import ChannelInsights, RawComment, StageASentimentItem, StageBClassificationItem
from storage.postgres import (
    claim_job,
    connect,
    create_job,
    finish_job,
    get_batch_results,
    get_completed_batch_keys,
    increment_completed_units,
    init_schema,
    is_cancel_requested,
    record_batch_result,
    request_cancellation,
    set_total_comment_count,
    update_job_progress,
)

logger = logging.getLogger(__name__)

STAGE_INGESTION = "ingestion"
STAGE_SENTIMENT = "stage_a_sentiment"
STAGE_EMBEDDING = "stage_a_embedding"
STAGE_CLASSIFY = "stage_b"
STAGE_INSIGHTS = "stage_c"
_INSIGHTS_BATCH_KEY = "insights"  # Stage C is checkpointed as a single unit -- see module docstring


class JobCancelledError(RuntimeError):
    """Raised to unwind a job's loop once cancellation is observed."""


@dataclass
class AnalysisResult:
    job_id: str
    channel_ref: str
    comments: list[RawComment]
    sentiments: dict[str, StageASentimentItem]
    embeddings: dict[str, list[float]]
    stage_b: dict[str, StageBClassificationItem]
    insights: ChannelInsights


async def _check_cancel(conn, job_id: str) -> None:
    if await is_cancel_requested(conn, job_id):
        raise JobCancelledError(f"Job {job_id} cancelled")


async def _run_ingestion(
    conn, job_id: str, estimate, *, client: httpx.AsyncClient, youtube_api_key: str,
    ledger: QuotaLedger, dedup: ContentDeduplicator,
) -> list[RawComment]:
    """One checkpoint unit = one video's comments."""
    done = await get_completed_batch_keys(conn, job_id, STAGE_INGESTION)
    cached = await get_batch_results(conn, job_id, STAGE_INGESTION)
    comments: list[RawComment] = []

    for video in estimate.videos:
        await _check_cancel(conn, job_id)
        if video.video_id in done:
            comments.extend(
                RawComment.model_validate(d) for d in cached[video.video_id]["comments"]
            )
            continue
        video_comments = (
            [
                c async for c in fetch_video_comments(
                    video.video_id, client=client, api_key=youtube_api_key,
                    ledger=ledger, dedup=dedup,
                )
            ]
            if video.comment_count > 0
            else []
        )
        await record_batch_result(
            conn, job_id, STAGE_INGESTION, video.video_id,
            result={"comments": [c.model_dump(mode="json") for c in video_comments]},
        )
        await increment_completed_units(conn, job_id, by=1)
        comments.extend(video_comments)
    return comments


async def _run_checkpointed_pydantic_batches(
    conn, job_id: str, stage: str, comments: list[RawComment], batch_size: int,
    *, item_model, run_batch,
) -> dict[str, object]:
    """Shared shape for Stage A sentiment and Stage B classification: chunk
    *comments* at batch_size, skip chunks already checkpointed, run the
    rest via run_batch(chunk) -> {comment_id: item}, checkpoint each new
    chunk, and return the merged (already-done + new) dict.
    """
    done = await get_completed_batch_keys(conn, job_id, stage)
    cached = await get_batch_results(conn, job_id, stage)
    results: dict[str, object] = {
        raw["comment_id"]: item_model.model_validate(raw)
        for payload in cached.values()
        for raw in payload["items"]
    }

    for start in range(0, len(comments), batch_size):
        batch_key = str(start)
        await _check_cancel(conn, job_id)
        if batch_key in done:
            continue
        batch = comments[start : start + batch_size]
        batch_result = await run_batch(batch)
        await record_batch_result(
            conn, job_id, stage, batch_key,
            result={"items": [item.model_dump(mode="json") for item in batch_result.values()]},
        )
        await increment_completed_units(conn, job_id, by=1)
        results.update(batch_result)
    return results


async def _run_stage_a_embeddings(
    conn, job_id: str, comments: list[RawComment], *, client: httpx.AsyncClient, api_key: str,
) -> dict[str, list[float]]:
    done = await get_completed_batch_keys(conn, job_id, STAGE_EMBEDDING)
    cached = await get_batch_results(conn, job_id, STAGE_EMBEDDING)
    results: dict[str, list[float]] = {}
    for payload in cached.values():
        results.update(payload["vectors"])

    for start in range(0, len(comments), DEFAULT_EMBEDDING_BATCH_SIZE):
        batch_key = str(start)
        await _check_cancel(conn, job_id)
        if batch_key in done:
            continue
        batch = comments[start : start + DEFAULT_EMBEDDING_BATCH_SIZE]
        vectors = await embed_comments_batch(batch, client=client, api_key=api_key)
        await record_batch_result(conn, job_id, STAGE_EMBEDDING, batch_key, result={"vectors": vectors})
        await increment_completed_units(conn, job_id, by=1)
        results.update(vectors)
    return results


def _n_batches(n_items: int, batch_size: int) -> int:
    return -(-n_items // batch_size) if n_items else 0


async def _run_stage_c(
    conn, job_id: str, comments, sentiments, stage_b_results, embeddings, video_titles,
    *, api_key: str,
) -> ChannelInsights:
    """Stage C, checkpointed as one unit (see module docstring)."""
    done = await get_completed_batch_keys(conn, job_id, STAGE_INSIGHTS)
    if _INSIGHTS_BATCH_KEY in done:
        cached = await get_batch_results(conn, job_id, STAGE_INSIGHTS)
        return ChannelInsights.model_validate(cached[_INSIGHTS_BATCH_KEY])

    result = await build_channel_insights(
        comments, sentiments, stage_b_results, embeddings, video_titles, api_key=api_key,
    )
    await record_batch_result(
        conn, job_id, STAGE_INSIGHTS, _INSIGHTS_BATCH_KEY, result=result.model_dump(mode="json"),
    )
    await increment_completed_units(conn, job_id, by=1)
    return result


async def analyze_channel(
    job_id: str,
    channel_ref: str,
    *,
    youtube_api_key: str,
    openrouter_api_key: str,
    max_videos: int = DEFAULT_MAX_VIDEOS,
) -> AnalysisResult:
    """SPEC §8: `analyze_channel(channel_id) -> AnalysisResult`, a plain
    async function composing the pipeline.

    Does not claim or finish the job itself — call via
    `run_job_in_background`/`start_analysis` for the full lifecycle
    (atomic claim, status transitions on completion/failure/cancellation).
    Calling this directly is useful for tests and for driving a job
    synchronously without the background-thread machinery.
    """
    async with connect() as conn:
        await init_schema(conn)
        ledger = QuotaLedger()
        dedup = ContentDeduplicator()

        async with httpx.AsyncClient() as client:
            estimate = await estimate_channel_analysis(
                channel_ref, client=client, api_key=youtube_api_key,
                ledger=ledger, max_videos=max_videos,
            )
            await update_job_progress(
                conn, job_id, stage=STAGE_INGESTION, total_units=len(estimate.videos),
                completed_units=0,
            )
            comments = await _run_ingestion(
                conn, job_id, estimate, client=client, youtube_api_key=youtube_api_key,
                ledger=ledger, dedup=dedup,
            )
            # SPEC §4.2 cache key's other half (channel_ref is set at
            # create_job time) -- lets a later analysis of this channel at
            # the same comment count find and reuse this job instead of
            # spending quota again (storage.postgres.find_reusable_job).
            await set_total_comment_count(conn, job_id, len(comments))

            await _check_cancel(conn, job_id)
            await update_job_progress(
                conn, job_id, stage=STAGE_SENTIMENT,
                total_units=_n_batches(len(comments), DEFAULT_SENTIMENT_BATCH_SIZE),
                completed_units=0,
            )
            sentiments = await _run_checkpointed_pydantic_batches(
                conn, job_id, STAGE_SENTIMENT, comments, DEFAULT_SENTIMENT_BATCH_SIZE,
                item_model=StageASentimentItem,
                run_batch=lambda batch: classify_sentiment_batch(batch, api_key=openrouter_api_key),
            )

            await _check_cancel(conn, job_id)
            await update_job_progress(
                conn, job_id, stage=STAGE_EMBEDDING,
                total_units=_n_batches(len(comments), DEFAULT_EMBEDDING_BATCH_SIZE),
                completed_units=0,
            )
            embeddings = await _run_stage_a_embeddings(
                conn, job_id, comments, client=client, api_key=openrouter_api_key
            )

            await _check_cancel(conn, job_id)
            flagged = select_for_stage_b(comments, sentiments, embeddings)
            await update_job_progress(
                conn, job_id, stage=STAGE_CLASSIFY,
                total_units=_n_batches(len(flagged), STAGE_B_BATCH_SIZE),
                completed_units=0,
            )
            stage_b_results = await _run_checkpointed_pydantic_batches(
                conn, job_id, STAGE_CLASSIFY, flagged, STAGE_B_BATCH_SIZE,
                item_model=StageBClassificationItem,
                run_batch=lambda batch: stage_b_classify_batch(batch, api_key=openrouter_api_key),
            )

            await _check_cancel(conn, job_id)
            video_titles = {v.video_id: v.title for v in estimate.videos}
            await update_job_progress(conn, job_id, stage=STAGE_INSIGHTS, total_units=1, completed_units=0)
            insights = await _run_stage_c(
                conn, job_id, comments, sentiments, stage_b_results, embeddings, video_titles,
                api_key=openrouter_api_key,
            )

    return AnalysisResult(
        job_id=job_id, channel_ref=channel_ref, comments=comments,
        sentiments=sentiments, embeddings=embeddings, stage_b=stage_b_results,
        insights=insights,
    )


async def _run_claimed_job(
    job_id: str, channel_ref: str, *, youtube_api_key: str, openrouter_api_key: str,
) -> None:
    async with connect() as conn:
        await init_schema(conn)
        claimed = await claim_job(conn, job_id)
    if not claimed:
        logger.info("Job %s already claimed/running/finished elsewhere; skipping", job_id)
        return

    try:
        await analyze_channel(
            job_id, channel_ref,
            youtube_api_key=youtube_api_key, openrouter_api_key=openrouter_api_key,
        )
    except JobCancelledError:
        async with connect() as conn:
            await finish_job(conn, job_id, status="cancelled")
    except Exception as exc:  # noqa: BLE001 -- a job must never stay stuck 'running'
        logger.exception("Job %s failed", job_id)
        async with connect() as conn:
            await finish_job(conn, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        async with connect() as conn:
            await finish_job(conn, job_id, status="completed")


def run_job_in_background(
    job_id: str, channel_ref: str, *, youtube_api_key: str, openrouter_api_key: str,
) -> threading.Thread:
    """SPEC §8: "Background execution: threading.Thread + a job-status row
    in Postgres." Returns the (already-started) thread; callers generally
    don't need to join it — poll storage.postgres.get_job_progress instead.
    """

    def _run() -> None:
        asyncio.run(_run_claimed_job(
            job_id, channel_ref,
            youtube_api_key=youtube_api_key, openrouter_api_key=openrouter_api_key,
        ))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


async def start_analysis(
    channel_ref: str, *, youtube_api_key: str, openrouter_api_key: str,
) -> str:
    """Create the job row and launch its background thread. Returns the
    job_id for the caller (app.py, Phase 8) to poll via
    storage.postgres.get_job_progress.
    """
    async with connect() as conn:
        await init_schema(conn)
        job_id = await create_job(conn, channel_ref)
    run_job_in_background(
        job_id, channel_ref,
        youtube_api_key=youtube_api_key, openrouter_api_key=openrouter_api_key,
    )
    return job_id


async def cancel_analysis(job_id: str) -> None:
    async with connect() as conn:
        await request_cancellation(conn, job_id)
