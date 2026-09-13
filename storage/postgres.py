"""SPEC §4.3/§9 — Postgres persistence, and the SPEC §8 orchestration
primitives that live on top of it: jobs/batches schema, atomic claim,
checkpointing, cancellation, progress query.

"One connection string, results survive redeploys" (§4.3) is the whole
point: this module talks to whatever `DATABASE_URL` points at, whether
that's this sandbox's own local Postgres (development) or Neon/Supabase
(production) — same schema, same driver (asyncpg), no code change between
the two.

Why atomic claim matters even in a single Streamlit process (SPEC §8):
"Streamlit Community Cloud sleeps inactive apps and the thread dies with
the container." On wake, nothing guarantees the old background thread is
really gone before a new one gets started for the same job — `claim_job`
makes sure only one of them can ever transition a job out of 'pending'.

Connections are opened per call/per job rather than pooled globally:
asyncpg connections are bound to the event loop that created them, and
each background job thread runs its own `asyncio.run(...)` with its own
fresh loop (SPEC §8: "Background execution: threading.Thread"). A shared
pool created on one loop can't be reused from another.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import asyncpg

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel_ref          TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'pending',
    stage                TEXT,
    total_units          INTEGER,
    completed_units      INTEGER NOT NULL DEFAULT 0,
    total_comment_count  INTEGER,
    cancel_requested     BOOLEAN NOT NULL DEFAULT FALSE,
    error                TEXT,
    claimed_at           TIMESTAMPTZ,
    started_at           TIMESTAMPTZ,
    finished_at          TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent migration for installations created before total_comment_count
-- existed (this sandbox's own dev database, notably).
ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS total_comment_count INTEGER;

CREATE TABLE IF NOT EXISTS analysis_job_batches (
    job_id       UUID NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    batch_key    TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'completed',
    result       JSONB,
    error        TEXT,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (job_id, stage, batch_key)
);

-- SPEC §4.2: "Cache by video, not by request... key on (video_id,
-- comment_count)." Approximated at channel granularity here (one job
-- already covers a whole channel's videos): the most recent completed job
-- for the same channel_ref with the same total_comment_count is reused
-- instead of spending quota again.
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_cache_lookup
    ON analysis_jobs (channel_ref, total_comment_count, status, created_at DESC);

-- Evaluation-criteria support ("Metrics Usage"): the last few times someone
-- clicked "Run live accuracy benchmark" in the app (app.py), so the result
-- survives a page reload/restart instead of living only in one Streamlit
-- session's memory. `metrics` is the same dict evals/benchmark.py already
-- produces and would otherwise only write to data/eval_metrics.json.
CREATE TABLE IF NOT EXISTS model_eval_runs (
    id         BIGSERIAL PRIMARY KEY,
    metrics    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _with_default_sslmode(dsn: str) -> str:
    """Add `sslmode=prefer` unless the DSN already specifies one.

    Different hosts this same DATABASE_URL might point at disagree on SSL:
    this sandbox's local dev Postgres doesn't offer it, Neon/Supabase
    require it, and a Render Postgres's behavior differs by which network
    path reaches it (this deploy's own external-network query tool hit a
    hard "SSL/TLS required" error that the app's internal-network
    connection may not). `prefer` negotiates SSL when the server offers
    it and falls back to plaintext when it doesn't, so the same connection
    string works unmodified against all of them rather than betting on
    which one applies.
    """
    parts = urllib.parse.urlsplit(dsn)
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    if "sslmode" in query:
        return dsn
    query["sslmode"] = ["prefer"]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunsplit(parts._replace(query=new_query))


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set (SPEC §4.3: Postgres/Neon for anything "
            "you must not lose). See .env.example."
        )
    return _with_default_sslmode(url)


@asynccontextmanager
async def connect() -> AsyncIterator[asyncpg.Connection]:
    """One short-lived connection. Callers doing several operations in a
    row (a job's whole run) should hold one `async with connect() as conn`
    for the duration rather than reconnecting per query."""
    conn = await asyncpg.connect(database_url())
    try:
        yield conn
    finally:
        await conn.close()


async def init_schema(conn: asyncpg.Connection) -> None:
    """Idempotent — safe to call at the start of every job (and in tests)."""
    await conn.execute(_SCHEMA_DDL)


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

async def create_job(conn: asyncpg.Connection, channel_ref: str) -> str:
    row = await conn.fetchrow(
        "INSERT INTO analysis_jobs (channel_ref) VALUES ($1) RETURNING id",
        channel_ref,
    )
    return str(row["id"])


async def claim_job(conn: asyncpg.Connection, job_id: str) -> bool:
    """Atomically transition *job_id* from 'pending' to 'running'.

    Returns True if this call performed the transition — the caller now
    owns running this job. Returns False if it was already claimed,
    running, or finished by someone else; the caller must not proceed.

    A single `UPDATE ... WHERE status = 'pending'` is atomic under
    Postgres: only one concurrent caller's statement can match a given row
    while it is still 'pending', so no explicit `SELECT ... FOR UPDATE` or
    application-level lock is needed.
    """
    result = await conn.execute(
        """
        UPDATE analysis_jobs
        SET status = 'running',
            claimed_at = now(),
            started_at = COALESCE(started_at, now()),
            updated_at = now()
        WHERE id = $1 AND status = 'pending'
        """,
        job_id,
    )
    return result == "UPDATE 1"


async def finish_job(
    conn: asyncpg.Connection, job_id: str, *, status: JobStatus, error: str | None = None
) -> None:
    await conn.execute(
        """
        UPDATE analysis_jobs
        SET status = $2, error = $3, finished_at = now(), updated_at = now()
        WHERE id = $1
        """,
        job_id, status, error,
    )


async def request_cancellation(conn: asyncpg.Connection, job_id: str) -> None:
    await conn.execute(
        "UPDATE analysis_jobs SET cancel_requested = TRUE, updated_at = now() WHERE id = $1",
        job_id,
    )


async def is_cancel_requested(conn: asyncpg.Connection, job_id: str) -> bool:
    row = await conn.fetchrow(
        "SELECT cancel_requested FROM analysis_jobs WHERE id = $1", job_id
    )
    return bool(row["cancel_requested"]) if row is not None else False


async def update_job_progress(
    conn: asyncpg.Connection,
    job_id: str,
    *,
    stage: str | None = None,
    total_units: int | None = None,
    completed_units: int | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE analysis_jobs
        SET stage = COALESCE($2, stage),
            total_units = COALESCE($3, total_units),
            completed_units = COALESCE($4, completed_units),
            updated_at = now()
        WHERE id = $1
        """,
        job_id, stage, total_units, completed_units,
    )


async def increment_completed_units(
    conn: asyncpg.Connection, job_id: str, by: int = 1
) -> None:
    await conn.execute(
        "UPDATE analysis_jobs SET completed_units = completed_units + $2, "
        "updated_at = now() WHERE id = $1",
        job_id, by,
    )


async def set_total_comment_count(conn: asyncpg.Connection, job_id: str, n: int) -> None:
    """SPEC §4.2's cache key's other half (channel_ref is the first) — set
    once ingestion knows the real count, so a later analysis of the same
    channel can tell whether anything changed."""
    await conn.execute(
        "UPDATE analysis_jobs SET total_comment_count = $2, updated_at = now() WHERE id = $1",
        job_id, n,
    )


async def find_reusable_job(
    conn: asyncpg.Connection, channel_ref: str, total_comment_count: int
) -> str | None:
    """SPEC §4.2: "Cache by video, not by request... If the comment count
    hasn't moved, serve the cached analysis." The most recent *completed*
    job for this exact (channel_ref, total_comment_count) pair, or None if
    nothing matches — meaning either this channel was never analyzed, or
    its comment count has moved since the last analysis.
    """
    row = await conn.fetchrow(
        """
        SELECT id FROM analysis_jobs
        WHERE channel_ref = $1 AND total_comment_count = $2 AND status = 'completed'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        channel_ref, total_comment_count,
    )
    return str(row["id"]) if row is not None else None


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    channel_ref: str
    status: JobStatus
    stage: str | None
    total_units: int | None
    completed_units: int
    total_comment_count: int | None
    cancel_requested: bool
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @property
    def fraction_complete(self) -> float | None:
        if not self.total_units:
            return None
        return min(1.0, self.completed_units / self.total_units)


async def get_job_progress(conn: asyncpg.Connection, job_id: str) -> JobProgress | None:
    row = await conn.fetchrow("SELECT * FROM analysis_jobs WHERE id = $1", job_id)
    if row is None:
        return None
    return JobProgress(
        job_id=str(row["id"]),
        channel_ref=row["channel_ref"],
        status=row["status"],
        stage=row["stage"],
        total_units=row["total_units"],
        completed_units=row["completed_units"],
        total_comment_count=row["total_comment_count"],
        cancel_requested=row["cancel_requested"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


# ---------------------------------------------------------------------------
# Checkpointing (SPEC §8: "write each completed batch to Postgres as it
# finishes ... the job function must be idempotent")
# ---------------------------------------------------------------------------

async def record_batch_result(
    conn: asyncpg.Connection,
    job_id: str,
    stage: str,
    batch_key: str,
    *,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    """Checkpoint one completed (or failed) unit of work. Idempotent: an
    already-recorded (job_id, stage, batch_key) is overwritten in place,
    so replaying a batch after a crash never creates a duplicate row.
    """
    status = "failed" if error else "completed"
    await conn.execute(
        """
        INSERT INTO analysis_job_batches (job_id, stage, batch_key, status, result, error, completed_at)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, now())
        ON CONFLICT (job_id, stage, batch_key)
        DO UPDATE SET status = EXCLUDED.status, result = EXCLUDED.result,
                      error = EXCLUDED.error, completed_at = now()
        """,
        job_id, stage, batch_key, status,
        json.dumps(result) if result is not None else None,
        error,
    )


async def get_completed_batch_keys(
    conn: asyncpg.Connection, job_id: str, stage: str
) -> set[str]:
    """Which batch_keys already have a completed checkpoint for this job +
    stage — SPEC §8: "On start, load completed batches and skip them."
    """
    rows = await conn.fetch(
        "SELECT batch_key FROM analysis_job_batches "
        "WHERE job_id = $1 AND stage = $2 AND status = 'completed'",
        job_id, stage,
    )
    return {r["batch_key"] for r in rows}


async def get_batch_results(
    conn: asyncpg.Connection, job_id: str, stage: str
) -> dict[str, dict]:
    """Every completed checkpoint's stored result for this job + stage,
    keyed by batch_key -- lets a resumed job rebuild state from Postgres
    instead of only knowing *that* a batch is done."""
    rows = await conn.fetch(
        "SELECT batch_key, result FROM analysis_job_batches "
        "WHERE job_id = $1 AND stage = $2 AND status = 'completed' AND result IS NOT NULL",
        job_id, stage,
    )
    return {r["batch_key"]: json.loads(r["result"]) for r in rows}


async def get_stage_checkpoint_summary(
    conn: asyncpg.Connection, job_id: str
) -> dict[str, dict]:
    """Per-stage checkpoint count and latest `completed_at` for *job_id* --
    the raw material app.py's `stage_durations_seconds` uses to derive an
    approximate stage-by-stage timing breakdown for the "Real-Time
    Efficiency" chart. Not a dedicated per-stage timer (SPEC §8 only
    requires a checkpoint per unit of work, not a profiler) — just what
    those checkpoints' own timestamps already tell us for free.
    """
    rows = await conn.fetch(
        """
        SELECT stage, MAX(completed_at) AS last_at, COUNT(*) AS n
        FROM analysis_job_batches
        WHERE job_id = $1 AND status = 'completed'
        GROUP BY stage
        """,
        job_id,
    )
    return {r["stage"]: {"last_at": r["last_at"], "n": r["n"]} for r in rows}


# ---------------------------------------------------------------------------
# Model evaluation runs ("Metrics Usage" — see the module's schema comment)
# ---------------------------------------------------------------------------

async def save_eval_run(conn: asyncpg.Connection, metrics: dict) -> None:
    await conn.execute(
        "INSERT INTO model_eval_runs (metrics) VALUES ($1::jsonb)",
        json.dumps(metrics),
    )


async def get_latest_eval_run(conn: asyncpg.Connection) -> dict | None:
    row = await conn.fetchrow(
        "SELECT metrics FROM model_eval_runs ORDER BY created_at DESC LIMIT 1"
    )
    return json.loads(row["metrics"]) if row is not None else None
