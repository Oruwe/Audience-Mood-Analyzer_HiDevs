"""Embedded DuckDB persistence for enriched analysis records."""

import asyncio
import json
import time
from pathlib import Path

import duckdb

from schemas import EnrichedCommentRecord

DB_PATH = Path("data/analytics.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyzed_comments (
    comment_id            VARCHAR PRIMARY KEY,
    platform              VARCHAR,
    author_handle         VARCHAR,
    raw_text              VARCHAR,
    sentiment             VARCHAR,
    confidence            DOUBLE,
    primary_intent        VARCHAR,
    urgency_score         DOUBLE,
    emotional_drivers     VARCHAR,
    summary               VARCHAR,
    recommended_action    VARCHAR,
    suggested_reply_draft VARCHAR,
    brand_safety_flag     BOOLEAN,
    embedding             VARCHAR,
    cluster_id            BIGINT,
    latency_ms            DOUBLE,
    model_used            VARCHAR,
    processed_at          TIMESTAMPTZ
);
"""

_INSERT_SQL = ("INSERT OR REPLACE INTO analyzed_comments VALUES "
               "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")

_COLUMNS = [
    "comment_id", "platform", "author_handle", "raw_text",
    "sentiment", "confidence", "primary_intent", "urgency_score",
    "emotional_drivers", "summary", "recommended_action",
    "suggested_reply_draft", "brand_safety_flag", "embedding",
    "cluster_id", "latency_ms", "model_used", "processed_at",
]

_LOCK_RETRIES = 6
_LOCK_BACKOFF_SEC = 0.5


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'analyzed_comments'"
    ).fetchone()[0]
    if exists:
        has_new = conn.execute(
            "SELECT COUNT(*) FROM duckdb_columns() "
            "WHERE table_name = 'analyzed_comments' AND column_name = 'sentiment'"
        ).fetchone()[0]
        if not has_new:
            conn.execute(
                "ALTER TABLE analyzed_comments RENAME TO analyzed_comments_legacy_v1"
            )
    conn.execute(_SCHEMA)


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


def insert_enriched_record(record: EnrichedCommentRecord) -> None:
    with _connect_write() as conn:
        conn.execute(_INSERT_SQL, [
            record.comment_id,
            record.platform,
            record.author_handle,
            record.raw_text,
            record.sentiment.value,
            record.confidence,
            record.primary_intent.value,
            record.urgency_score,
            json.dumps(record.emotional_drivers),
            record.summary,
            record.recommended_action.value,
            record.suggested_reply_draft,
            record.brand_safety_flag,
            json.dumps(record.embedding) if record.embedding is not None else None,
            record.cluster_id,
            record.latency_ms,
            record.model_used,
            record.processed_at,
        ])


def _rows_to_records(rows: list[tuple]) -> list[EnrichedCommentRecord]:
    records: list[EnrichedCommentRecord] = []
    for row in rows:
        data = dict(zip(_COLUMNS, row))
        data["emotional_drivers"] = (
            json.loads(data["emotional_drivers"]) if data["emotional_drivers"] else []
        )
        data["embedding"] = (
            json.loads(data["embedding"]) if data["embedding"] else None
        )
        records.append(EnrichedCommentRecord(**data))
    return records


def query_enriched_records(limit: int | None = None) -> list[EnrichedCommentRecord]:
    sql = "SELECT * FROM analyzed_comments ORDER BY processed_at DESC"
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect_read() as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rows_to_records(rows)


_ESCALATION_ACTIONS = ("escalate_to_support", "escalate_to_pr")
_URGENT_THRESHOLD = 0.8


def get_top_urgent_escalations(limit: int = 5) -> list[EnrichedCommentRecord]:
    """Highest-urgency rows needing escalation, newest first."""
    sql = (
        "SELECT * FROM analyzed_comments "
        f"WHERE urgency_score >= {_URGENT_THRESHOLD} "
        f"OR recommended_action IN {repr(_ESCALATION_ACTIONS)} "
        "ORDER BY urgency_score DESC, processed_at DESC LIMIT ?"
    )
    with _connect_read() as conn:
        rows = conn.execute(sql, [limit]).fetchall()
    return _rows_to_records(rows)


async def aget_top_urgent_escalations(limit: int = 5) -> list[EnrichedCommentRecord]:
    return await asyncio.to_thread(get_top_urgent_escalations, limit)


async def ainsert_analyzed_mood(record: EnrichedCommentRecord) -> None:
    """Alias kept for Phase 3 API compatibility."""
    await ainsert_enriched_record(record)


async def ainsert_enriched_record(record: EnrichedCommentRecord) -> None:
    await asyncio.to_thread(insert_enriched_record, record)


async def aquery_enriched_records(limit: int | None = None) -> list[EnrichedCommentRecord]:
    return await asyncio.to_thread(query_enriched_records, limit)


def close_connection() -> None:
    """No-op kept for API compatibility — connections are now short-lived."""
