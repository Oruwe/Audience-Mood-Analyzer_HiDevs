"""L2 contract test — SPEC §10 invariant 5: tenant isolation.

"Tenant isolation — every query scoped by user_id, parameterized."
Proven by: "Integration test: user A's session cannot read user B's row."

This file was a `pytest.mark.skip` scaffold until the `owner_id` column
landed in storage/postgres.py. It is now a real integration test against a
real Postgres (gated by the `pg_dsn` fixture like the rest of the DB suite):
it writes two tenants' jobs and checkpoints, then asserts that every read
path reachable with a caller-supplied job_id returns nothing when the owner
does not match.

The threat model is deliberately the pessimistic one. These tests hand user A
user B's *exact* job_id -- the UUID is assumed already leaked or guessed, so
what is being measured is the SQL `owner_id` predicate and nothing else. An
assertion that passed only because UUIDs are hard to guess would be testing
luck, not the invariant.

Uses asyncio.run() directly, like the rest of this repo's async tests, rather
than adding an async-test plugin.
"""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from storage.postgres import (
    DEFAULT_OWNER_ID,
    claim_job,
    create_job,
    find_reusable_job,
    get_batch_results,
    get_completed_batch_keys,
    get_job_progress,
    get_stage_checkpoint_summary,
    init_schema,
    is_cancel_requested,
    record_batch_result,
    request_cancellation,
    set_total_comment_count,
    update_job_progress,
)

USER_A = "user-a@example.com"
USER_B = "user-b@example.com"

B_SECRET = "user B's private analysis"


def _run(coro):
    return asyncio.run(coro)


async def _open(pg_dsn: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(pg_dsn)
    await init_schema(conn)
    return conn


async def _job_owned_by_b(conn: asyncpg.Connection) -> str:
    """A job owned by user B, carrying a checkpoint with recognisable
    content, so a leak shows up as B's actual data and not just a row count.
    """
    job_id = await create_job(conn, "channel-of-user-b", owner_id=USER_B)
    await record_batch_result(
        conn, job_id, "stage_c", "batch-0", result={"secret": B_SECRET},
    )
    await update_job_progress(
        conn, job_id, stage="stage_c", total_units=1, completed_units=1
    )
    return job_id


def test_user_a_session_cannot_read_user_b_row(pg_dsn):
    """The headline assertion SPEC §10 names: A holds B's job_id and still
    cannot read the row."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            mine = await get_job_progress(conn, job_id, owner_id=USER_B)
            theirs = await get_job_progress(conn, job_id, owner_id=USER_A)
            return mine, theirs
        finally:
            await conn.close()

    mine, theirs = _run(scenario())
    assert mine is not None, "precondition: B must be able to read B's own row"
    assert theirs is None, "user A read user B's job row"


def test_user_a_cannot_read_user_b_checkpoint_results(pg_dsn):
    """The job row is metadata; the checkpoints hold the analysed content.
    This is the leak that would actually matter."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            mine = await get_batch_results(conn, job_id, "stage_c", owner_id=USER_B)
            theirs = await get_batch_results(conn, job_id, "stage_c", owner_id=USER_A)
            return mine, theirs
        finally:
            await conn.close()

    mine, theirs = _run(scenario())
    assert mine["batch-0"]["secret"] == B_SECRET
    assert theirs == {}, f"user A read user B's analysis: {theirs!r}"


def test_user_a_cannot_enumerate_user_b_batch_keys(pg_dsn):
    """Even without the payload, the key list leaks how much B analysed."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            return (
                await get_completed_batch_keys(conn, job_id, "stage_c", owner_id=USER_B),
                await get_completed_batch_keys(conn, job_id, "stage_c", owner_id=USER_A),
            )
        finally:
            await conn.close()

    mine, theirs = _run(scenario())
    assert mine == {"batch-0"}
    assert theirs == set()


def test_user_a_cannot_read_user_b_stage_timings(pg_dsn):
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            return (
                await get_stage_checkpoint_summary(conn, job_id, owner_id=USER_B),
                await get_stage_checkpoint_summary(conn, job_id, owner_id=USER_A),
            )
        finally:
            await conn.close()

    mine, theirs = _run(scenario())
    assert mine != {}
    assert theirs == {}


def test_cache_does_not_serve_one_tenant_the_others_analysis(pg_dsn):
    """The reuse cache is the subtlest crossing: it is keyed on channel_ref,
    and two tenants analysing the same public channel is the normal case, not
    an edge case. Without owner scoping, A would silently be handed B's
    completed job -- a leak that never looks like an error, because serving a
    cached analysis is exactly what the cache is supposed to do.
    """
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await create_job(conn, "shared-public-channel", owner_id=USER_B)
            await set_total_comment_count(conn, job_id, 500)
            await conn.execute(
                "UPDATE analysis_jobs SET status = 'completed' WHERE id = $1", job_id
            )
            return (
                job_id,
                await find_reusable_job(conn, "shared-public-channel", 500, owner_id=USER_B),
                await find_reusable_job(conn, "shared-public-channel", 500, owner_id=USER_A),
            )
        finally:
            await conn.close()

    job_id, mine, theirs = _run(scenario())
    assert mine == job_id, "B must still get B's own cache hit"
    assert theirs is None, "user A was served user B's cached analysis"


def test_user_a_cannot_cancel_user_b_job(pg_dsn):
    """Isolation is not only about reads. An unscoped cancel is a
    cross-tenant denial of service."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            await request_cancellation(conn, job_id, owner_id=USER_A)
            after_foreign = await is_cancel_requested(conn, job_id)
            await request_cancellation(conn, job_id, owner_id=USER_B)
            after_owner = await is_cancel_requested(conn, job_id)
            return after_foreign, after_owner
        finally:
            await conn.close()

    after_foreign, after_owner = _run(scenario())
    assert after_foreign is False, "user A cancelled user B's job"
    assert after_owner is True, "user B could not cancel their own job"


def test_user_a_cannot_claim_user_b_job(pg_dsn):
    """Claiming someone else's pending job would run it under A's worker and
    write the results into B's rows."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            return (
                await claim_job(conn, job_id, owner_id=USER_A),
                await claim_job(conn, job_id, owner_id=USER_B),
            )
        finally:
            await conn.close()

    by_a, by_b = _run(scenario())
    assert by_a is False, "user A claimed user B's job"
    assert by_b is True, "user B could not claim their own job"


def test_default_owner_is_a_real_tenant_not_a_wildcard(pg_dsn):
    """DEFAULT_OWNER_ID must behave like any other owner value. If it were
    treated as "unscoped", the single-tenant deployment would run a code path
    no test covers, and any forgotten owner_id argument would fail open --
    which is the failure mode this whole invariant exists to prevent.
    """
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await _job_owned_by_b(conn)
            return (
                await get_job_progress(conn, job_id, owner_id=DEFAULT_OWNER_ID),
                await get_batch_results(conn, job_id, "stage_c", owner_id=DEFAULT_OWNER_ID),
            )
        finally:
            await conn.close()

    progress, batches = _run(scenario())
    assert progress is None, "DEFAULT_OWNER_ID acted as a wildcard over job rows"
    assert batches == {}, "DEFAULT_OWNER_ID acted as a wildcard over checkpoints"


def test_owner_id_column_rejects_null(pg_dsn):
    """A NULL owner would be a row belonging to nobody -- and under SQL's
    three-valued logic, `owner_id = $1` never matches it, so such a row would
    become permanently unreadable rather than safely private. Either way the
    schema must forbid it."""
    async def scenario():
        conn = await _open(pg_dsn)
        try:
            job_id = await create_job(conn, "chan", owner_id=USER_A)
            with pytest.raises(asyncpg.NotNullViolationError):
                await conn.execute(
                    "UPDATE analysis_jobs SET owner_id = NULL WHERE id = $1", job_id
                )
        finally:
            await conn.close()

    _run(scenario())
