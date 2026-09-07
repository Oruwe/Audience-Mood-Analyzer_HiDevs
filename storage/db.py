"""Embedded DuckDB persistence for enriched analysis records."""

import ast
import asyncio
import json
import logging
import time
from pathlib import Path

import duckdb

from schemas import EnrichedCommentRecord

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "analytics.duckdb"

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


def _missing_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    present = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM duckdb_columns() "
            "WHERE table_name = 'analyzed_comments'"
        ).fetchall()
    }
    return [col for col in _COLUMNS if col not in present]


def _backfill_from_legacy(conn: duckdb.DuckDBPyConnection, legacy_name: str) -> None:
    """Carry overlapping columns from *legacy_name* into the fresh table.

    Columns introduced since the legacy snapshot (e.g. ``embedding``) are
    filled with NULL — downstream consumers already tolerate missing vectors
    and cluster ids. On any failure the brand-new table stays empty and the
    archive is preserved for manual recovery, matching the old behaviour.
    """
    try:
        present = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM duckdb_columns() "
                f"WHERE table_name = '{legacy_name}'"
            ).fetchall()
        }
        select_parts = [f'"{col}"' if col in present else "NULL" for col in _COLUMNS]
        carried = [col for col in _COLUMNS if col in present]
        inserted = int(conn.execute(
            f"INSERT INTO analyzed_comments "
            f"SELECT {', '.join(select_parts)} FROM {legacy_name}"
        ).fetchone()[0])
        logger.info(
            "schema migration: carried %d row(s) forward from %s (columns: %s)",
            inserted, legacy_name, ", ".join(carried) or "none",
        )
    except Exception as exc:  # noqa: BLE001 — startup must survive a bad legacy file
        logger.warning(
            "auto-backfill from %s failed (%s); starting empty — "
            "archive table kept for manual recovery",
            legacy_name, exc,
        )


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'analyzed_comments'"
    ).fetchone()[0]
    if exists and _missing_columns(conn):
        # Stale pre-embedding layout — archive it, recreate fresh, then pull
        # every still-compatible column across so history survives upgrades.
        # Timestamped suffix keeps repeat migrations collision-free.
        legacy_name = f"analyzed_comments_legacy_{int(time.time())}"
        conn.execute(f"ALTER TABLE analyzed_comments RENAME TO {legacy_name}")
        conn.execute(_SCHEMA)
        _backfill_from_legacy(conn, legacy_name)
        return
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


def _migrate_stale_file() -> None:
    """Upgrade a legacy warehouse before read-only access.

    Readers open the file read-only and cannot ALTER anything, so a stale
    file would crash the dashboard indefinitely. Probe with a read-only
    connection; if columns are missing, briefly take the write lock (which
    runs _ensure_schema) to archive + recreate + backfill. Best-effort: if
    the pipeline holds the write lock, skip quietly and let the normal read
    path proceed.
    """
    if not DB_PATH.exists():
        return  # preserve the callers' FileNotFoundError behaviour
    try:
        with _connect_read() as conn:
            if not _missing_columns(conn):
                return
    except Exception:  # noqa: BLE001 — locked/unreadable; read path will report it
        return
    try:
        with _connect_write():
            pass  # _ensure_schema inside _connect_write performs the migration
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not migrate legacy warehouse (%s); read may fail", exc)


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


def _parse_jsonish(raw: object, fallback: object) -> object:
    """Tolerant column decoder: JSON first, Python-literal second, else fallback."""
    if raw is None or isinstance(raw, (list, dict)):
        return raw  # NULL, or a native DuckDB LIST/STRUCT already materialised
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(raw)  # handles "None", "['a', 'b']", "(1, 2)"
        except Exception:
            logger.warning("unparseable column value %r — using fallback", str(raw)[:60])
            return fallback


def _rows_to_records(rows: list[tuple]) -> list[EnrichedCommentRecord]:
    records: list[EnrichedCommentRecord] = []
    for row in rows:
        data = dict(zip(_COLUMNS, row, strict=True))
        data["emotional_drivers"] = _parse_jsonish(data["emotional_drivers"], [])
        data["embedding"] = _parse_jsonish(data["embedding"], None)
        records.append(EnrichedCommentRecord(**data))
    return records


def query_enriched_records(limit: int | None = None) -> list[EnrichedCommentRecord]:
    _migrate_stale_file()
    sql = "SELECT * FROM analyzed_comments ORDER BY processed_at DESC"
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with _connect_read() as conn:
        rows = conn.execute(sql, params).fetchall()
    return _rows_to_records(rows)


def get_comments_since(minutes: int) -> list[EnrichedCommentRecord]:
    """All enriched rows processed within the trailing N-minute window."""
    _migrate_stale_file()
    sql = ("SELECT * FROM analyzed_comments "
           "WHERE processed_at >= current_timestamp - to_minutes(?) "
           "ORDER BY processed_at DESC")
    with _connect_read() as conn:
        rows = conn.execute(sql, [minutes]).fetchall()
    return _rows_to_records(rows)


async def aget_comments_since(minutes: int) -> list[EnrichedCommentRecord]:
    return await asyncio.to_thread(get_comments_since, minutes)


_ESCALATION_ACTIONS = ("escalate_to_support", "escalate_to_pr")
_URGENT_THRESHOLD = 0.8


def get_top_urgent_escalations(limit: int = 5) -> list[EnrichedCommentRecord]:
    """Highest-urgency rows needing escalation, newest first."""
    _migrate_stale_file()
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
