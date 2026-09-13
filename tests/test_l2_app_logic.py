"""L2 tests — app.py's async logic layer.

`preflight`/`launch_analysis` are tested against mocks (they just delegate
to ingestion.youtube/orchestration, both already covered thoroughly by
their own test suites — this file checks the *wiring*, not re-testing
those internals). `find_cached_analysis`/`load_progress`/`load_insights`
are tested against this sandbox's real local Postgres (tests/conftest.py's
`pg_dsn`, skips cleanly if unreachable) since they're thin reads over it.
"""

import asyncio

import asyncpg
import pytest

import app
from ingestion.youtube import QuotaLedger
from schemas import ChannelInsights
from storage.postgres import create_job, finish_job, init_schema, record_batch_result, set_total_comment_count


def _run(coro):
    return asyncio.run(coro)


def test_preflight_delegates_to_estimate_channel_analysis_with_given_key_and_ledger(monkeypatch):
    captured = {}

    async def fake_estimate(channel_ref, *, client, api_key, ledger):
        captured.update(channel_ref=channel_ref, api_key=api_key, ledger=ledger)
        return "sentinel-estimate"

    monkeypatch.setattr(app, "estimate_channel_analysis", fake_estimate)
    ledger = QuotaLedger()

    result = _run(app.preflight("chan", youtube_api_key="yt-key", ledger=ledger))

    assert result == "sentinel-estimate"
    assert captured["channel_ref"] == "chan"
    assert captured["api_key"] == "yt-key"
    assert captured["ledger"] is ledger


def test_launch_analysis_delegates_to_orchestration_start_analysis(monkeypatch):
    captured = {}

    async def fake_start_analysis(channel_ref, *, youtube_api_key, openrouter_api_key):
        captured.update(
            channel_ref=channel_ref, youtube_api_key=youtube_api_key,
            openrouter_api_key=openrouter_api_key,
        )
        return "job-123"

    monkeypatch.setattr(app, "start_analysis", fake_start_analysis)

    job_id = _run(app.launch_analysis("chan", youtube_api_key="yt", openrouter_api_key="or"))

    assert job_id == "job-123"
    assert captured == {
        "channel_ref": "chan", "youtube_api_key": "yt", "openrouter_api_key": "or",
    }


def test_find_cached_analysis_finds_a_completed_job_with_matching_count(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            job_id = await create_job(conn, "chan-app-test")
            await set_total_comment_count(conn, job_id, 7)
            await finish_job(conn, job_id, status="completed")
        finally:
            await conn.close()
        found = await app.find_cached_analysis("chan-app-test", 7)
        not_found_count = await app.find_cached_analysis("chan-app-test", 999)
        not_found_channel = await app.find_cached_analysis("some-other-chan", 7)
        return job_id, found, not_found_count, not_found_channel

    job_id, found, not_found_count, not_found_channel = _run(scenario())
    assert found == job_id
    assert not_found_count is None
    assert not_found_channel is None


def test_load_progress_reads_a_real_job(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            job_id = await create_job(conn, "chan")
        finally:
            await conn.close()
        return job_id, await app.load_progress(job_id)

    job_id, progress = _run(scenario())
    assert progress.job_id == job_id
    assert progress.status == "pending"


def test_load_progress_returns_none_for_an_unknown_job(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    result = _run(app.load_progress("00000000-0000-0000-0000-000000000000"))
    assert result is None


def test_load_insights_reads_the_checkpointed_stage_c_result(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    expected = ChannelInsights(requests=[], confusion_points=[], video_moods=[])

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            job_id = await create_job(conn, "chan")
            await record_batch_result(
                conn, job_id, "stage_c", "insights", result=expected.model_dump(mode="json")
            )
        finally:
            await conn.close()
        return job_id, await app.load_insights(job_id)

    job_id, insights = _run(scenario())
    assert insights == expected


def test_load_insights_returns_none_before_stage_c_checkpoints(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            return await create_job(conn, "chan")
        finally:
            await conn.close()

    job_id = _run(scenario())
    assert _run(app.load_insights(job_id)) is None
