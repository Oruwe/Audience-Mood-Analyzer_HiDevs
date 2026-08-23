"""Embedded DuckDB persistence for analysed comments."""

import asyncio
import time
from pathlib import Path
import duckdb
from schemas import AnalyzedMood, RawComment

DB_PATH = Path("data/analytics.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyzed_comments (
    comment_id       VARCHAR PRIMARY KEY,
    platform         VARCHAR,
    author           VARCHAR,
    comment_text     VARCHAR,
    mood             VARCHAR,
    confidence       DOUBLE,
    summary          VARCHAR,
    urgency_score    DOUBLE,
    marketing_action VARCHAR,
    processed_at     TIMESTAMPTZ
);
"""

# Upgrades DBs created before raw-comment columns existed (idempotent).
_MIGRATIONS = [
    "ALTER TABLE analyzed_comments ADD COLUMN IF NOT EXISTS platform VARCHAR",
    "ALTER TABLE analyzed_comments ADD COLUMN IF NOT EXISTS author VARCHAR",
    "ALTER TABLE analyzed_comments ADD COLUMN IF NOT EXISTS comment_text VARCHAR",
]

_INSERT_SQL = "INSERT OR REPLACE INTO analyzed_comments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

_COLUMNS = ["comment_id", "platform", "author", "comment_text", "mood",
            "confidence", "summary", "urgency_score", "marketing_action", "processed_at"]

_LOCK_RETRIES = 6
_LOCK_BACKOFF_SEC = 0.5


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(_SCHEMA)
    for stmt in _MIGRATIONS:
        conn.execute(stmt)


def _connect_write() -> duckdb.DuckDBPyConnection:
    """Short-lived RW connection; retries while another process holds the lock."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for _ in range(_LOCK_RETRIES):
        try:
            conn = duckdb.connect(str(DB_PATH))
            _ensure_schema(conn)
            return conn
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(_LOCK_BACKOFF_SEC)
    raise last_exc  # type: ignore[misc]


def _connect_read() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"{DB_PATH} does not exist yet")
    last_exc: Exception | None = None
    for _ in range(_LOCK_RETRIES):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except duckdb.IOException as exc:
            last_exc = exc
            time.sleep(_LOCK_BACKOFF_SEC)
    raise last_exc  # type: ignore[misc]


def insert_analyzed_mood(mood: AnalyzedMood, comment: RawComment) -> None:
    with _connect_write() as conn:
        conn.execute(_INSERT_SQL, [
            mood.comment_id, comment.platform, comment.author, comment.text,
            mood.mood.value, mood.confidence, mood.summary,
            mood.urgency_score, mood.marketing_action.value, mood.processed_at,
        ])


def query_analyzed_comments(limit: int | None = None) -> list[AnalyzedMood]:
    sql = "SELECT * FROM analyzed_comments ORDER BY processed_at DESC"
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect_read() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [AnalyzedMood(**dict(zip(_COLUMNS, row))) for row in rows]


async def ainsert_analyzed_mood(mood: AnalyzedMood, comment: RawComment) -> None:
    await asyncio.to_thread(insert_analyzed_mood, mood, comment)


async def aquery_analyzed_comments(limit: int | None = None) -> list[AnalyzedMood]:
    return await asyncio.to_thread(query_analyzed_comments, limit)


def close_connection() -> None:
    """No-op kept for API compatibility — connections are now short-lived."""
