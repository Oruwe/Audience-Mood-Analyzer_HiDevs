"""L2/L3 integration tests — orchestration.py (SPEC §8) against a real
Postgres (tests/conftest.py's `pg_dsn`, skips cleanly if unreachable), with
ingestion/Stage A/Stage B mocked out (no network, no API keys) so these
tests exercise the orchestration wiring itself: checkpointing, resume,
cancellation, atomic claim, and progress reporting.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import orchestration
from ingestion.youtube import ChannelInfo, QuotaEstimate, VideoMeta
from schemas import (
    ChannelInsights,
    CommentIntent,
    RawComment,
    Sentiment,
    StageASentimentItem,
    StageBClassificationItem,
)
from storage.postgres import get_job_progress, init_schema
import asyncpg


def _estimate(video_specs: list[tuple[str, int]]) -> QuotaEstimate:
    videos = [VideoMeta(video_id=vid, title=f"Video {vid}", comment_count=n) for vid, n in video_specs]
    return QuotaEstimate(
        channel=ChannelInfo(channel_id="UCabc", title="Test Channel",
                             uploads_playlist_id="UUabc", video_count=len(videos)),
        videos=videos,
        total_comment_count=sum(n for _, n in video_specs),
        units_already_spent_on_estimate=1,
        units_required_for_comment_pull=len(videos),
    )


def _comment(video_id: str, i: int) -> RawComment:
    return RawComment(
        id=f"{video_id}-c{i}", platform="youtube", text=f"comment {i} on {video_id}",
        timestamp=datetime.now(timezone.utc), video_id=video_id,
    )


def _install_happy_path_mocks(monkeypatch, *, video_specs, calls):
    async def fake_estimate(channel_ref, **kwargs):
        calls.setdefault("estimate", 0)
        calls["estimate"] += 1
        return _estimate(video_specs)

    async def fake_fetch_video_comments(video_id, **kwargs):
        calls.setdefault("fetch_video", []).append(video_id)
        n = dict(video_specs)[video_id]
        for i in range(n):
            yield _comment(video_id, i)

    async def fake_classify_sentiment_batch(batch, *, api_key, model=None):
        calls.setdefault("sentiment_batches", []).append([c.id for c in batch])
        return {
            c.id: StageASentimentItem(comment_id=c.id, sentiment=Sentiment.NEUTRAL, confidence=0.9)
            for c in batch
        }

    async def fake_embed_comments_batch(batch, *, client, api_key, model=None):
        calls.setdefault("embed_batches", []).append([c.id for c in batch])
        return {c.id: [float(i), 0.0] for i, c in enumerate(batch)}

    async def fake_stage_b_classify_batch(batch, *, api_key, model=None):
        calls.setdefault("stage_b_batches", []).append([c.id for c in batch])
        return {
            c.id: StageBClassificationItem(
                comment_id=c.id, intent=CommentIntent.OTHER, is_request=False, is_confusion=False
            )
            for c in batch
        }

    async def fake_build_channel_insights(comments, sentiments, stage_b, embeddings, video_titles, *, api_key, model=None):
        calls.setdefault("insights_calls", 0)
        calls["insights_calls"] += 1
        return ChannelInsights(requests=[], confusion_points=[], video_moods=[])

    monkeypatch.setattr(orchestration, "estimate_channel_analysis", fake_estimate)
    monkeypatch.setattr(orchestration, "fetch_video_comments", fake_fetch_video_comments)
    monkeypatch.setattr(orchestration, "classify_sentiment_batch", fake_classify_sentiment_batch)
    monkeypatch.setattr(orchestration, "embed_comments_batch", fake_embed_comments_batch)
    monkeypatch.setattr(orchestration, "stage_b_classify_batch", fake_stage_b_classify_batch)
    monkeypatch.setattr(orchestration, "build_channel_insights", fake_build_channel_insights)


def _run(coro):
    return asyncio.run(coro)


def test_analyze_channel_happy_path(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    video_specs = [("v1", 2), ("v2", 3)]
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()

        result = await orchestration.analyze_channel(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )
        return job_id, result

    job_id, result = _run(scenario())

    assert len(result.comments) == 5
    assert set(result.sentiments.keys()) == {c.id for c in result.comments}
    assert set(result.embeddings.keys()) == {c.id for c in result.comments}
    # all comments are high-confidence (0.9) -> all flagged for Stage B
    assert set(result.stage_b.keys()) == {c.id for c in result.comments}
    assert calls["fetch_video"] == ["v1", "v2"]
    assert isinstance(result.insights, ChannelInsights)
    assert calls["insights_calls"] == 1


def test_resuming_skips_already_checkpointed_ingestion(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    video_specs = [("v1", 2), ("v2", 3)]
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()

        # First run: full pipeline.
        await orchestration.analyze_channel(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )
        first_fetch_calls = list(calls["fetch_video"])
        first_sentiment_calls = len(calls["sentiment_batches"])

        # Second run, same job_id: everything should be a checkpoint hit.
        await orchestration.analyze_channel(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )
        return first_fetch_calls, first_sentiment_calls

    first_fetch_calls, first_sentiment_calls = _run(scenario())

    assert first_fetch_calls == ["v1", "v2"]
    # No new fetch_video_comments calls on the second run -- ingestion was
    # fully checkpointed.
    assert calls["fetch_video"] == ["v1", "v2"]
    # No new sentiment batches either.
    assert len(calls["sentiment_batches"]) == first_sentiment_calls
    # Stage C (checkpointed as a single unit) also isn't recomputed.
    assert calls["insights_calls"] == 1


def test_cancellation_stops_the_job_between_stages(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    video_specs = [("v1", 2)]
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job, request_cancellation
            job_id = await create_job(conn, "chan")
            # Cancel before the job even starts -- must stop at the very
            # first cancellation checkpoint, before any ingestion happens.
            await request_cancellation(conn, job_id)
        finally:
            await conn.close()

        with pytest.raises(orchestration.JobCancelledError):
            await orchestration.analyze_channel(
                job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
            )
        return job_id

    _run(scenario())
    assert "fetch_video" not in calls  # never got past the first cancel check


def test_run_claimed_job_marks_completed_on_success(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=[("v1", 1)], calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()

        await orchestration._run_claimed_job(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            return await get_job_progress(conn2, job_id)
        finally:
            await conn2.close()

    progress = _run(scenario())
    assert progress.status == "completed"
    assert progress.finished_at is not None


def test_run_claimed_job_marks_failed_on_exception(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def boom(*args, **kwargs):
        raise RuntimeError("YouTube is on fire")

    monkeypatch.setattr(orchestration, "estimate_channel_analysis", boom)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()

        await orchestration._run_claimed_job(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            return await get_job_progress(conn2, job_id)
        finally:
            await conn2.close()

    progress = _run(scenario())
    assert progress.status == "failed"
    assert "YouTube is on fire" in progress.error


def test_run_claimed_job_is_a_no_op_if_already_claimed(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=[("v1", 1)], calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import claim_job, create_job
            job_id = await create_job(conn, "chan")
            # Pre-claim the job, simulating another runner already owning it.
            already_claimed = await claim_job(conn, job_id)
            assert already_claimed is True
        finally:
            await conn.close()

        # _run_claimed_job must see status != 'pending' and refuse to run
        # the pipeline at all.
        await orchestration._run_claimed_job(
            job_id, "chan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )
        return job_id

    _run(scenario())
    assert "fetch_video" not in calls


def test_start_analysis_and_cancel_analysis_wire_through(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=[("v1", 1)], calls=calls)

    # Avoid actually spawning a background thread in this test -- just
    # prove start_analysis creates the row and returns a usable job_id, and
    # cancel_analysis flips the flag storage.postgres already tests directly.
    monkeypatch.setattr(orchestration, "run_job_in_background", lambda *a, **kw: None)

    async def scenario():
        job_id = await orchestration.start_analysis(
            "https://youtube.com/@testchan", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )
        await orchestration.cancel_analysis(job_id)

        conn = await asyncpg.connect(pg_dsn)
        try:
            return await get_job_progress(conn, job_id)
        finally:
            await conn.close()

    progress = _run(scenario())
    assert progress.status == "pending"  # background thread never actually ran
    assert progress.cancel_requested is True


def test_analyze_channel_records_total_comment_count_for_the_spec_4_2_cache(monkeypatch, pg_dsn):
    """SPEC §4.2: "cache by video, not by request... if the comment count
    hasn't moved, serve the cached analysis." Confirms analyze_channel sets
    the count find_reusable_job later looks up by."""
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=[("v1", 2), ("v2", 3)], calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job, find_reusable_job
            job_id = await create_job(conn, "chan-for-cache-test")
        finally:
            await conn.close()

        # _run_claimed_job (not analyze_channel directly) so the job
        # actually lands on 'completed' -- find_reusable_job only matches
        # completed jobs.
        await orchestration._run_claimed_job(
            job_id, "chan-for-cache-test", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            progress = await get_job_progress(conn2, job_id)
            found = await find_reusable_job(conn2, "chan-for-cache-test", 5)
            not_found = await find_reusable_job(conn2, "chan-for-cache-test", 999)
            return progress, found, not_found, job_id
        finally:
            await conn2.close()

    progress, found, not_found, job_id = _run(scenario())
    assert progress.total_comment_count == 5  # 2 + 3 comments across the two videos
    assert found == job_id
    assert not_found is None


def test_multiple_concurrent_batches_all_checkpoint_correctly(monkeypatch, pg_dsn):
    """_run_checkpointed_pydantic_batches/_run_stage_a_embeddings now fire
    every not-yet-done batch concurrently (asyncio.gather) rather than one
    at a time, with Postgres writes serialized behind an asyncio.Lock
    since they share one asyncpg connection. This forces enough comments
    to actually produce multiple concurrent Stage A sentiment/embedding
    batches (DEFAULT_SENTIMENT_BATCH_SIZE=50, DEFAULT_EMBEDDING_BATCH_SIZE=
    100) against a real Postgres, to catch a lock/interleaving bug the
    single-batch tests above can't -- e.g. a dropped or duplicated
    checkpoint row, or a corrupted completed_units count.
    """
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    # 3 videos x 40 comments = 120 total -> 3 sentiment batches (50/50/20)
    # and 2 embedding batches (100/20), all genuinely concurrent.
    video_specs = [("v1", 40), ("v2", 40), ("v3", 40)]
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan-concurrency-test")
        finally:
            await conn.close()

        result = await orchestration.analyze_channel(
            job_id, "chan-concurrency-test", youtube_api_key="yt-key", openrouter_api_key="or-key",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            progress = await get_job_progress(conn2, job_id)
            from storage.postgres import get_completed_batch_keys
            sentiment_keys = await get_completed_batch_keys(conn2, job_id, orchestration.STAGE_SENTIMENT)
            embedding_keys = await get_completed_batch_keys(conn2, job_id, orchestration.STAGE_EMBEDDING)
        finally:
            await conn2.close()
        return result, progress, sentiment_keys, embedding_keys

    result, progress, sentiment_keys, embedding_keys = _run(scenario())

    assert len(result.comments) == 120
    assert set(result.sentiments.keys()) == {c.id for c in result.comments}
    assert set(result.embeddings.keys()) == {c.id for c in result.comments}
    assert sentiment_keys == {"0", "50", "100"}  # 3 concurrent batches, none dropped/duplicated
    assert embedding_keys == {"0", "100"}        # 2 concurrent batches
    # completed_units accumulated correctly despite concurrent
    # increment_completed_units calls sharing one connection under the lock
    # (3 sentiment + 2 embedding + N stage_b + 1 stage_c batches by the
    # time Stage C's own update_job_progress resets the counter for its
    # stage -- checked directly against the call counts instead, which
    # don't depend on which stage's counter is currently live).
    assert len(calls["sentiment_batches"]) == 3
    assert len(calls["embed_batches"]) == 2
    assert progress.stage == orchestration.STAGE_INSIGHTS


# ---------------------------------------------------------------------------
# Concurrency cap, resume-progress accounting, and the job deadline.
#
# All three are fixes for defects found auditing before the first paid run:
# unbounded gather (SPEC §8 named a Semaphore; it had been reasoned away on
# the false premise that batch counts are small), a progress bar that could
# never reach 100% on a resumed job, and a wedged job that could hold
# 'running' forever because cancellation is cooperative and a hung await
# never reaches the next check.
# ---------------------------------------------------------------------------

def test_bounded_gather_never_exceeds_the_concurrency_cap():
    peak = {"now": 0, "max": 0}

    async def work():
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        await asyncio.sleep(0.01)
        peak["now"] -= 1
        return 1

    results = _run(orchestration._bounded_gather([work() for _ in range(50)], limit=4))

    assert sum(results) == 50      # every unit of work still ran
    assert peak["max"] <= 4        # but never more than 4 at once


def test_bounded_gather_preserves_result_order():
    async def work(i):
        # Later items finish first, so ordering can only be right if
        # gather's ordering guarantee is preserved through the semaphore.
        await asyncio.sleep(0.01 * (5 - i))
        return i

    assert _run(orchestration._bounded_gather([work(i) for i in range(5)], limit=2)) == [0, 1, 2, 3, 4]


def test_resumed_job_progress_counts_work_already_checkpointed(monkeypatch, pg_dsn):
    """A resumed job skips checkpointed batches, so if the counter started
    at 0 it could never reach total -- the bar would sit at 2/5 forever on
    a job that was actually finished."""
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    video_specs = [("v1", 40), ("v2", 40), ("v3", 40)]  # 120 comments -> 3 sentiment batches
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan-resume-progress")
        finally:
            await conn.close()

        # First run completes everything and checkpoints it.
        await orchestration.analyze_channel(
            job_id, "chan-resume-progress", youtube_api_key="yt", openrouter_api_key="or",
        )
        first_pass_batches = len(calls["sentiment_batches"])

        # Second run over the same job: every batch is a checkpoint hit, so
        # nothing increments -- progress must still report fully complete.
        await orchestration.analyze_channel(
            job_id, "chan-resume-progress", youtube_api_key="yt", openrouter_api_key="or",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            return first_pass_batches, len(calls["sentiment_batches"]), await get_job_progress(conn2, job_id)
        finally:
            await conn2.close()

    first_pass, second_pass, progress = _run(scenario())

    assert first_pass == 3
    assert second_pass == 3               # no re-work on resume
    assert progress.total_units == 1      # last stage (stage_c) is a single unit
    assert progress.completed_units == 1  # and it is counted as done, not 0
    assert progress.fraction_complete == 1.0


def test_a_wedged_job_hits_its_deadline_and_is_marked_failed(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setattr(orchestration, "JOB_DEADLINE_SECONDS", 0.2)

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(orchestration, "analyze_channel", never_returns)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan-wedged")
        finally:
            await conn.close()

        await orchestration._run_claimed_job(
            job_id, "chan-wedged", youtube_api_key="yt", openrouter_api_key="or",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            return await get_job_progress(conn2, job_id)
        finally:
            await conn2.close()

    progress = _run(scenario())
    assert progress.status == "failed"
    assert "limit" in (progress.error or "")
    assert progress.finished_at is not None


def test_a_running_job_heartbeats_so_the_reaper_leaves_it_alone(monkeypatch, pg_dsn):
    """The heartbeat task must actually beat during a real run -- otherwise
    reap_stale_jobs would retire live jobs mid-analysis."""
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=[("v1", 2)], calls=calls)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan-heartbeat")
        finally:
            await conn.close()

        await orchestration._run_claimed_job(
            job_id, "chan-heartbeat", youtube_api_key="yt", openrouter_api_key="or",
        )

        conn2 = await asyncpg.connect(pg_dsn)
        try:
            return await get_job_progress(conn2, job_id)
        finally:
            await conn2.close()

    progress = _run(scenario())
    assert progress.status == "completed"
    # Claim + every checkpointed batch touches it, so it is set well before
    # the periodic task's first tick.
    assert progress.heartbeat_at is not None


# ---------------------------------------------------------------------------
# Ingestion concurrency.
#
# This is the regression that cost a live analysis its entire 15-minute
# deadline (2026-09-14, job 3e998fba). Ingestion was the only serial stage in
# the pipeline, so it was also the only one whose cost scaled with channel
# size -- and because it makes no LLM calls, it failed by being slow rather
# than by raising, leaving no warnings and no errors to diagnose from.
#
# Asserting "it finished" would not catch a return to sequential: the
# sequential version finished too, just far too slowly. So these assert the
# property that actually differs -- overlap -- and the ordering guarantee the
# rewrite had to preserve while gaining it.
# ---------------------------------------------------------------------------

def test_ingestion_fetches_videos_concurrently(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    video_specs = [(f"v{i}", 1) for i in range(8)]
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    in_flight = 0
    peak_in_flight = 0

    async def slow_fetch(video_id, **kwargs):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await asyncio.sleep(0.05)  # a network round-trip's worth of yielding
            yield _comment(video_id, 0)
        finally:
            in_flight -= 1

    monkeypatch.setattr(orchestration, "fetch_video_comments", slow_fetch)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()
        return await orchestration.analyze_channel(
            job_id, "chan", youtube_api_key="yt", openrouter_api_key="or",
        )

    _run(scenario())

    # Sequential ingestion never exceeds 1. The exact peak depends on
    # scheduling, so this asserts overlap happened at all rather than
    # pinning a number.
    assert peak_in_flight > 1, (
        f"ingestion ran {peak_in_flight} video(s) at a time — it is sequential again"
    )
    assert peak_in_flight <= orchestration.MAX_CONCURRENT_BATCHES


def test_ingestion_returns_comments_in_channel_order_not_completion_order(monkeypatch, pg_dsn):
    """Concurrency must not reorder the corpus.

    Downstream batch keys are positional slices of this list, and a resumed
    job rebuilds it from a mix of checkpointed and freshly-fetched videos. If
    completion order leaked through, a resume would produce a different
    corpus from the run it is resuming.
    """
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    video_specs = [("v0", 1), ("v1", 1), ("v2", 1)]
    calls: dict = {}
    _install_happy_path_mocks(monkeypatch, video_specs=video_specs, calls=calls)

    # Finish in reverse order: v2 first, v0 last.
    delays = {"v0": 0.06, "v1": 0.03, "v2": 0.0}

    async def staggered_fetch(video_id, **kwargs):
        await asyncio.sleep(delays[video_id])
        yield _comment(video_id, 0)

    monkeypatch.setattr(orchestration, "fetch_video_comments", staggered_fetch)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            from storage.postgres import create_job
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()
        return await orchestration.analyze_channel(
            job_id, "chan", youtube_api_key="yt", openrouter_api_key="or",
        )

    result = _run(scenario())

    assert [c.video_id for c in result.comments] == ["v0", "v1", "v2"]
