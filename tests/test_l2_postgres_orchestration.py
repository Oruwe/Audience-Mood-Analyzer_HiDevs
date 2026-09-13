"""L2/L3 integration tests — storage.postgres (SPEC §8 orchestration
primitives) against a real Postgres, gated by the `pg_dsn` fixture
(tests/conftest.py — skips cleanly if none is reachable).

Uses asyncio.run() directly (like the rest of this repo's async tests)
rather than an async-test pytest plugin, to avoid adding one. The
atomic-claim test in particular can't be validated against a fake
connection: it's specifically about Postgres's own row-level atomicity
under concurrent access, so this is real integration testing, not a mock.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from storage.postgres import (
    claim_job,
    create_job,
    finish_job,
    get_batch_results,
    get_completed_batch_keys,
    get_job_progress,
    get_latest_eval_run,
    get_stage_checkpoint_summary,
    increment_completed_units,
    init_schema,
    is_cancel_requested,
    record_batch_result,
    request_cancellation,
    save_eval_run,
    update_job_progress,
)


def _run(coro):
    return asyncio.run(coro)


async def _fresh_job(pg_dsn: str, channel_ref: str = "chan") -> tuple[asyncpg.Connection, str]:
    """Open a connection, ensure the schema exists, and create one job.
    Caller is responsible for closing the returned connection."""
    conn = await asyncpg.connect(pg_dsn)
    await init_schema(conn)
    job_id = await create_job(conn, channel_ref)
    return conn, job_id


def test_init_schema_is_idempotent(pg_dsn):
    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            await init_schema(conn)  # second call, same connection -- must not raise
        finally:
            await conn.close()

    _run(scenario())


def test_create_job_starts_pending_with_zero_progress(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn, "https://youtube.com/@testchan")
        try:
            return await get_job_progress(conn, job_id)
        finally:
            await conn.close()

    progress = _run(scenario())
    assert progress is not None
    assert progress.status == "pending"
    assert progress.channel_ref == "https://youtube.com/@testchan"
    assert progress.completed_units == 0
    assert progress.cancel_requested is False
    assert progress.fraction_complete is None  # total_units not set yet


def test_get_job_progress_returns_none_for_unknown_job(pg_dsn):
    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            fake_id = "00000000-0000-0000-0000-000000000000"
            return await get_job_progress(conn, fake_id)
        finally:
            await conn.close()

    assert _run(scenario()) is None


def test_claim_job_succeeds_once_and_refuses_a_second_claim(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            first = await claim_job(conn, job_id)
            second = await claim_job(conn, job_id)
            progress = await get_job_progress(conn, job_id)
            return first, second, progress
        finally:
            await conn.close()

    first, second, progress = _run(scenario())
    assert first is True
    assert second is False
    assert progress.status == "running"
    assert progress.started_at is not None


def test_claim_job_is_atomic_under_real_concurrency(pg_dsn):
    """The actual guarantee: several independent connections racing to
    claim the same job -- exactly one must win, never both, never neither.
    """

    async def scenario():
        setup_conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(setup_conn)
            job_id = await create_job(setup_conn, "race-test-channel")
        finally:
            await setup_conn.close()

        async def attempt() -> bool:
            c = await asyncpg.connect(pg_dsn)
            try:
                return await claim_job(c, job_id)
            finally:
                await c.close()

        return await asyncio.gather(*[attempt() for _ in range(10)])

    results = _run(scenario())
    assert sum(results) == 1  # exactly one of ten concurrent claims wins


def test_finish_job_records_status_error_and_finished_at(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await finish_job(conn, job_id, status="failed", error="quota exceeded")
            return await get_job_progress(conn, job_id)
        finally:
            await conn.close()

    progress = _run(scenario())
    assert progress.status == "failed"
    assert progress.error == "quota exceeded"
    assert progress.finished_at is not None


def test_cancellation_flag_round_trips(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            before = await is_cancel_requested(conn, job_id)
            await request_cancellation(conn, job_id)
            after = await is_cancel_requested(conn, job_id)
            return before, after
        finally:
            await conn.close()

    before, after = _run(scenario())
    assert before is False
    assert after is True


def test_progress_tracking_and_fraction_complete(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await update_job_progress(conn, job_id, stage="stage_a", total_units=10)
            await increment_completed_units(conn, job_id, by=3)
            await increment_completed_units(conn, job_id, by=2)
            return await get_job_progress(conn, job_id)
        finally:
            await conn.close()

    progress = _run(scenario())
    assert progress.stage == "stage_a"
    assert progress.total_units == 10
    assert progress.completed_units == 5
    assert progress.fraction_complete == pytest.approx(0.5)


def test_checkpointing_is_idempotent_and_resumable(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await record_batch_result(conn, job_id, "ingestion", "video-1", result={"n": 5})
            await record_batch_result(conn, job_id, "ingestion", "video-2", result={"n": 7})
            completed = await get_completed_batch_keys(conn, job_id, "ingestion")

            # Re-recording the same key (a resumed/replayed job) must
            # overwrite in place, not create a duplicate or raise a PK error.
            await record_batch_result(conn, job_id, "ingestion", "video-1", result={"n": 999})
            results = await get_batch_results(conn, job_id, "ingestion")
            return completed, results
        finally:
            await conn.close()

    completed, results = _run(scenario())
    assert completed == {"video-1", "video-2"}
    assert results["video-1"] == {"n": 999}
    assert len(results) == 2


def test_failed_batch_is_not_counted_as_completed(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await record_batch_result(conn, job_id, "stage_b", "batch-0", error="all providers failed")
            return await get_completed_batch_keys(conn, job_id, "stage_b")
        finally:
            await conn.close()

    assert _run(scenario()) == set()


def test_batch_checkpoints_are_scoped_per_stage(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await record_batch_result(conn, job_id, "stage_a_sentiment", "batch-0", result={"ok": True})
            await record_batch_result(conn, job_id, "stage_b", "batch-0", result={"ok": True})
            a = await get_completed_batch_keys(conn, job_id, "stage_a_sentiment")
            b = await get_completed_batch_keys(conn, job_id, "stage_b")
            return a, b
        finally:
            await conn.close()

    a, b = _run(scenario())
    assert a == {"batch-0"}
    assert b == {"batch-0"}


# ---------------------------------------------------------------------------
# get_stage_checkpoint_summary / save_eval_run / get_latest_eval_run --
# evaluation-criteria support (Real-Time Efficiency's stage-timing chart,
# Metrics Usage's persisted accuracy benchmark).
# ---------------------------------------------------------------------------

def test_stage_checkpoint_summary_counts_and_latest_timestamp_per_stage(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            await record_batch_result(conn, job_id, "ingestion", "v1", result={"n": 1})
            await record_batch_result(conn, job_id, "ingestion", "v2", result={"n": 1})
            await record_batch_result(conn, job_id, "stage_c", "insights", result={"n": 1})
            # A failed checkpoint must not count toward the summary.
            await record_batch_result(conn, job_id, "stage_b", "batch-0", error="boom")
            return await get_stage_checkpoint_summary(conn, job_id)
        finally:
            await conn.close()

    summary = _run(scenario())
    assert summary["ingestion"]["n"] == 2
    assert summary["stage_c"]["n"] == 1
    assert "stage_b" not in summary
    assert summary["ingestion"]["last_at"] is not None
    assert summary["stage_c"]["last_at"] is not None


def test_stage_checkpoint_summary_empty_for_a_job_with_no_checkpoints(pg_dsn):
    async def scenario():
        conn, job_id = await _fresh_job(pg_dsn)
        try:
            return await get_stage_checkpoint_summary(conn, job_id)
        finally:
            await conn.close()

    assert _run(scenario()) == {}


def test_eval_run_round_trips_and_latest_wins(pg_dsn):
    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            await save_eval_run(conn, {"accuracy": 0.5, "generated_at": "t1"})
            await save_eval_run(conn, {"accuracy": 0.9, "generated_at": "t2"})
            return await get_latest_eval_run(conn)
        finally:
            await conn.close()

    latest = _run(scenario())
    assert latest["accuracy"] == 0.9
    assert latest["generated_at"] == "t2"


def test_get_latest_eval_run_none_when_table_is_empty(pg_dsn):
    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            await conn.execute("DELETE FROM model_eval_runs")
            return await get_latest_eval_run(conn)
        finally:
            await conn.close()

    assert _run(scenario()) is None
