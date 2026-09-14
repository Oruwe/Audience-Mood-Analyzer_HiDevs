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
from storage.postgres import (
    claim_job,
    create_job,
    finish_job,
    get_latest_eval_run,
    init_schema,
    record_batch_result,
    set_total_comment_count,
)


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


def test_load_sentiment_distribution_counts_a_real_checkpointed_stage(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            job_id = await create_job(conn, "chan")
            await record_batch_result(
                conn, job_id, "stage_a_sentiment", "0",
                result={"items": [
                    {"comment_id": "c0", "sentiment": "positive"},
                    {"comment_id": "c1", "sentiment": "positive"},
                    {"comment_id": "c2", "sentiment": "negative"},
                ]},
            )
        finally:
            await conn.close()
        return job_id, await app.load_sentiment_distribution(job_id)

    job_id, distribution = _run(scenario())
    assert distribution == {"positive": 2, "negative": 1}


def test_load_stage_durations_reads_real_checkpoints(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            job_id = await create_job(conn, "chan")
            await claim_job(conn, job_id)  # sets started_at -- stage_durations_seconds needs it
            await record_batch_result(conn, job_id, "ingestion", "v1", result={"n": 1})
            progress = await app.load_progress(job_id)
        finally:
            await conn.close()
        durations = await app.load_stage_durations(job_id, progress)
        return durations

    durations = _run(scenario())
    assert "ingestion" in durations
    assert durations["ingestion"] >= 0.0


def test_run_and_persist_eval_persists_a_stubbed_benchmark_result(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def fake_run_benchmark(*, persist_to_file):
        assert persist_to_file is False
        return {"accuracy": 0.75, "generated_at": "test-time"}

    # app.py imports evals.benchmark lazily (it drags in scikit-learn, which
    # does not fit in this service's 512 MiB alongside everything else), so
    # there is no `app.benchmark` attribute to patch — patch the module the
    # lazy import will resolve to.
    import evals.benchmark as benchmark
    monkeypatch.setattr(benchmark, "run_benchmark", fake_run_benchmark)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            await conn.execute("DELETE FROM model_eval_runs")
        finally:
            await conn.close()
        result = await app.run_and_persist_eval()
        loaded = await app.load_latest_eval()
        return result, loaded

    result, loaded = _run(scenario())
    assert result == {"accuracy": 0.75, "generated_at": "test-time"}
    assert loaded == {"accuracy": 0.75, "generated_at": "test-time"}


def test_load_latest_eval_none_when_nothing_has_run_yet(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    async def scenario():
        conn = await asyncpg.connect(pg_dsn)
        try:
            await init_schema(conn)
            await conn.execute("DELETE FROM model_eval_runs")
        finally:
            await conn.close()
        return await app.load_latest_eval()

    assert _run(scenario()) is None
    # get_latest_eval_run itself agrees -- not just app's wrapper.
    async def check_directly():
        conn = await asyncpg.connect(pg_dsn)
        try:
            return await get_latest_eval_run(conn)
        finally:
            await conn.close()
    assert _run(check_directly()) is None
