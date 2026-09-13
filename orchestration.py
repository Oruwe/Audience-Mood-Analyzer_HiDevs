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
    touch_heartbeat,
    update_job_progress,
)

logger = logging.getLogger(__name__)

STAGE_INGESTION = "ingestion"
STAGE_SENTIMENT = "stage_a_sentiment"
STAGE_EMBEDDING = "stage_a_embedding"
STAGE_CLASSIFY = "stage_b"
STAGE_INSIGHTS = "stage_c"
_INSIGHTS_BATCH_KEY = "insights"  # Stage C is checkpointed as a single unit -- see module docstring

# SPEC §8 named `asyncio.Semaphore` over batch calls as part of the
# concurrency design. When per-stage batches were first made concurrent
# this cap was left off, reasoned as "batch counts per stage are small and
# bounded by construction" -- which is wrong: batch count scales with a
# channel's comment count, so a large channel fires one request per 50
# comments *simultaneously*. At a few thousand comments that's ~100 open
# HTTPS requests at once, which trips provider rate limits that wouldn't
# otherwise fire, exhausts httpx's connection pool, and spikes memory on a
# 512 MB instance. 8 is comfortably above what's needed to hide per-call
# latency and comfortably below any of those ceilings.
MAX_CONCURRENT_BATCHES = 8

# Network timeouts for the shared httpx client (YouTube pagination and the
# OpenRouter embeddings endpoint). httpx's default is 5s for *every* phase,
# which is far too tight for an embeddings call carrying 100 comments --
# it turns a normal slow response into a spurious timeout, a retry, and a
# multiplied wall-clock cost. Connect stays short (a genuinely unreachable
# host should fail fast); read/write are generous.
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)

# A job that has not finished within this long is not "slow", it is stuck:
# no single analysis at this project's scale legitimately runs an hour, and
# an unbounded job can never be cancelled, reaped, or retried -- it just
# holds its row at 'running' forever. `_run_claimed_job` enforces it.
JOB_DEADLINE_SECONDS = 900  # 15 minutes

# How often the background heartbeat task marks a running job as alive.
# Must stay well under storage.postgres.reap_stale_jobs' staleness window.
HEARTBEAT_INTERVAL_SECONDS = 30


class JobCancelledError(RuntimeError):
    """Raised to unwind a job's loop once cancellation is observed."""


class JobDeadlineExceededError(RuntimeError):
    """The job exceeded JOB_DEADLINE_SECONDS and was abandoned."""


async def _bounded_gather(coros, *, limit: int = MAX_CONCURRENT_BATCHES):
    """`asyncio.gather`, but with at most *limit* of them in flight —
    SPEC §8's "asyncio.Semaphore over batch calls". See
    MAX_CONCURRENT_BATCHES for why the cap is not optional."""
    semaphore = asyncio.Semaphore(limit)

    async def _guarded(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_guarded(c) for c in coros))


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

    # Count work already checkpointed by an earlier attempt, so a resumed
    # job's progress bar starts where it left off instead of at 0/N and
    # never reaching N (it skips those batches, so nothing would ever
    # increment for them).
    already_done = sum(1 for video in estimate.videos if video.video_id in done)
    await update_job_progress(
        conn, job_id, stage=STAGE_INGESTION,
        total_units=len(estimate.videos), completed_units=already_done,
    )

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
        await touch_heartbeat(conn, job_id)
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

    Not-yet-checkpointed batches run concurrently (asyncio.gather) rather
    than one at a time -- the actual run_batch(...) calls (network/LLM,
    no shared state) are fully independent; only the Postgres writes that
    follow each one share *conn* (one asyncpg connection per job, per this
    module's docstring), which asyncpg doesn't allow concurrent queries
    on, so those are serialized behind `_db_lock`. Cancellation is
    therefore checked once, before firing this stage's whole batch of
    calls, rather than between every individual batch as before -- a
    coarser granularity (a cancel now takes effect between stages'
    concurrent waves, not between every single batch within one), traded
    for the latency win of not waiting on batches sequentially.
    """
    done = await get_completed_batch_keys(conn, job_id, stage)
    cached = await get_batch_results(conn, job_id, stage)
    results: dict[str, object] = {
        raw["comment_id"]: item_model.model_validate(raw)
        for payload in cached.values()
        for raw in payload["items"]
    }

    all_keys = [str(start) for start in range(0, len(comments), batch_size)]
    pending = [
        (key, comments[int(key) : int(key) + batch_size])
        for key in all_keys
        if key not in done
    ]
    # Resumed jobs skip already-checkpointed batches, so seed the counter
    # with them rather than starting at 0 and never reaching total.
    await update_job_progress(
        conn, job_id, stage=stage,
        total_units=len(all_keys), completed_units=len(all_keys) - len(pending),
    )

    await _check_cancel(conn, job_id)
    db_lock = asyncio.Lock()

    async def _run_one(batch_key: str, batch: list[RawComment]) -> dict[str, object]:
        batch_result = await run_batch(batch)
        async with db_lock:
            await record_batch_result(
                conn, job_id, stage, batch_key,
                result={"items": [item.model_dump(mode="json") for item in batch_result.values()]},
            )
            await increment_completed_units(conn, job_id, by=1)
            await touch_heartbeat(conn, job_id)
        return batch_result

    for batch_result in await _bounded_gather(
        [_run_one(key, batch) for key, batch in pending]
    ):
        results.update(batch_result)
    return results


async def _run_stage_a_embeddings(
    conn, job_id: str, comments: list[RawComment], *, client: httpx.AsyncClient, api_key: str,
) -> dict[str, list[float]]:
    """Same concurrency/locking shape as `_run_checkpointed_pydantic_batches`
    -- see its docstring."""
    done = await get_completed_batch_keys(conn, job_id, STAGE_EMBEDDING)
    cached = await get_batch_results(conn, job_id, STAGE_EMBEDDING)
    results: dict[str, list[float]] = {}
    for payload in cached.values():
        results.update(payload["vectors"])

    all_keys = [str(start) for start in range(0, len(comments), DEFAULT_EMBEDDING_BATCH_SIZE)]
    pending = [
        (key, comments[int(key) : int(key) + DEFAULT_EMBEDDING_BATCH_SIZE])
        for key in all_keys
        if key not in done
    ]
    await update_job_progress(
        conn, job_id, stage=STAGE_EMBEDDING,
        total_units=len(all_keys), completed_units=len(all_keys) - len(pending),
    )

    await _check_cancel(conn, job_id)
    db_lock = asyncio.Lock()

    async def _run_one(batch_key: str, batch: list[RawComment]) -> dict[str, list[float]]:
        vectors = await embed_comments_batch(batch, client=client, api_key=api_key)
        async with db_lock:
            await record_batch_result(conn, job_id, STAGE_EMBEDDING, batch_key, result={"vectors": vectors})
            await increment_completed_units(conn, job_id, by=1)
            await touch_heartbeat(conn, job_id)
        return vectors

    for vectors in await _bounded_gather(
        [_run_one(key, batch) for key, batch in pending]
    ):
        results.update(vectors)
    return results


async def _run_stage_c(
    conn, job_id: str, comments, sentiments, stage_b_results, embeddings, video_titles,
    *, api_key: str,
) -> ChannelInsights:
    """Stage C, checkpointed as one unit (see module docstring)."""
    done = await get_completed_batch_keys(conn, job_id, STAGE_INSIGHTS)
    already_done = _INSIGHTS_BATCH_KEY in done
    # Set progress here rather than in the caller, for the same reason as
    # the other stages: only this function knows whether the single unit is
    # already checkpointed, and a resumed job that skips it would otherwise
    # report 0/1 forever.
    await update_job_progress(
        conn, job_id, stage=STAGE_INSIGHTS,
        total_units=1, completed_units=1 if already_done else 0,
    )
    if already_done:
        cached = await get_batch_results(conn, job_id, STAGE_INSIGHTS)
        return ChannelInsights.model_validate(cached[_INSIGHTS_BATCH_KEY])

    result = await build_channel_insights(
        comments, sentiments, stage_b_results, embeddings, video_titles, api_key=api_key,
    )
    await record_batch_result(
        conn, job_id, STAGE_INSIGHTS, _INSIGHTS_BATCH_KEY, result=result.model_dump(mode="json"),
    )
    await increment_completed_units(conn, job_id, by=1)
    await touch_heartbeat(conn, job_id)
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

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            estimate = await estimate_channel_analysis(
                channel_ref, client=client, api_key=youtube_api_key,
                ledger=ledger, max_videos=max_videos,
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
            sentiments = await _run_checkpointed_pydantic_batches(
                conn, job_id, STAGE_SENTIMENT, comments, DEFAULT_SENTIMENT_BATCH_SIZE,
                item_model=StageASentimentItem,
                run_batch=lambda batch: classify_sentiment_batch(batch, api_key=openrouter_api_key),
            )

            await _check_cancel(conn, job_id)
            embeddings = await _run_stage_a_embeddings(
                conn, job_id, comments, client=client, api_key=openrouter_api_key
            )

            await _check_cancel(conn, job_id)
            flagged = select_for_stage_b(comments, sentiments, embeddings)
            stage_b_results = await _run_checkpointed_pydantic_batches(
                conn, job_id, STAGE_CLASSIFY, flagged, STAGE_B_BATCH_SIZE,
                item_model=StageBClassificationItem,
                run_batch=lambda batch: stage_b_classify_batch(batch, api_key=openrouter_api_key),
            )

            await _check_cancel(conn, job_id)
            video_titles = {v.video_id: v.title for v in estimate.videos}
            insights = await _run_stage_c(
                conn, job_id, comments, sentiments, stage_b_results, embeddings, video_titles,
                api_key=openrouter_api_key,
            )

    return AnalysisResult(
        job_id=job_id, channel_ref=channel_ref, comments=comments,
        sentiments=sentiments, embeddings=embeddings, stage_b=stage_b_results,
        insights=insights,
    )


async def _heartbeat_loop(job_id: str, interval: float = HEARTBEAT_INTERVAL_SECONDS) -> None:
    """Mark *job_id* alive every *interval* seconds until cancelled.

    Runs on its own connection: the job's own connection is busy (and
    serialized behind a lock during concurrent stages), and asyncpg does
    not allow concurrent queries on one connection. Per-batch heartbeats
    alone aren't enough — a stage that hangs mid-call would stop
    heartbeating and get reaped as dead even though the process is fine.
    This ticks regardless of what the pipeline is doing, so the heartbeat
    means exactly one thing: *this process is still alive*. Detecting a
    process that is alive but wedged is JOB_DEADLINE_SECONDS' job, not
    this one's.
    """
    try:
        async with connect() as conn:
            while True:
                await asyncio.sleep(interval)
                await touch_heartbeat(conn, job_id)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 -- a failing heartbeat must never fail the job
        logger.warning("Heartbeat for job %s stopped early", job_id, exc_info=True)


async def _run_claimed_job(
    job_id: str, channel_ref: str, *, youtube_api_key: str, openrouter_api_key: str,
) -> None:
    async with connect() as conn:
        await init_schema(conn)
        claimed = await claim_job(conn, job_id)
        if claimed:
            await touch_heartbeat(conn, job_id)
    if not claimed:
        logger.info("Job %s already claimed/running/finished elsewhere; skipping", job_id)
        return

    heartbeat = asyncio.create_task(_heartbeat_loop(job_id))
    try:
        # A wedged job must fail, not hang: without this it holds 'running'
        # forever, uncancellable (cancellation is cooperative and a wedged
        # await never reaches the next check) and un-reapable (the
        # heartbeat above keeps proving the *process* is alive).
        await asyncio.wait_for(
            analyze_channel(
                job_id, channel_ref,
                youtube_api_key=youtube_api_key, openrouter_api_key=openrouter_api_key,
            ),
            timeout=JOB_DEADLINE_SECONDS,
        )
    except JobCancelledError:
        async with connect() as conn:
            await finish_job(conn, job_id, status="cancelled")
    except (TimeoutError, asyncio.TimeoutError):
        logger.error("Job %s exceeded its %ss deadline", job_id, JOB_DEADLINE_SECONDS)
        async with connect() as conn:
            await finish_job(
                conn, job_id, status="failed",
                error=(
                    f"Analysis exceeded its {JOB_DEADLINE_SECONDS // 60}-minute limit and was "
                    "stopped. Completed work was checkpointed — running the same analysis "
                    "again resumes from where it stopped."
                ),
            )
    except Exception as exc:  # noqa: BLE001 -- a job must never stay stuck 'running'
        logger.exception("Job %s failed", job_id)
        async with connect() as conn:
            await finish_job(conn, job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
    else:
        async with connect() as conn:
            await finish_job(conn, job_id, status="completed")
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass


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
