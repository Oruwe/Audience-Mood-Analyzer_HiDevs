"""Embedded DuckDB persistence for analysed comments."""

import asyncio
from pathlib import Path
import duckdb
from schemas import AnalyzedMood

DB_PATH = Path("data/analytics.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyzed_comments (
    comment_id       VARCHAR PRIMARY KEY,
    mood             VARCHAR,
    confidence       DOUBLE,
    summary          VARCHAR,
    urgency_score    DOUBLE,
    marketing_action VARCHAR,
    processed_at     TIMESTAMPTZ
);
"""

_INSERT_SQL = "INSERT OR REPLACE INTO analyzed_comments VALUES (?, ?, ?, ?, ?, ?, ?)"

_COLUMNS = ["comment_id", "mood", "confidence", "summary",
            "urgency_score", "marketing_action", "processed_at"]

_conn: duckdb.DuckDBPyConnection | None = None

def get_connection() -> duckdb.DuckDBPyConnection:
    """Lazily create the embedded DB file, its parent dir, and the table."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
        _conn.execute(_SCHEMA)
    return _conn

def insert_analyzed_mood(mood: AnalyzedMood) -> None:
    get_connection().execute(_INSERT_SQL, [
        mood.comment_id, mood.mood.value, mood.confidence, mood.summary,
        mood.urgency_score, mood.marketing_action.value, mood.processed_at,
    ])

def query_analyzed_comments(limit: int | None = None) -> list[AnalyzedMood]:
    sql = "SELECT * FROM analyzed_comments ORDER BY processed_at DESC"
    params: list = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = get_connection().execute(sql, params).fetchall()
    return [AnalyzedMood(**dict(zip(_COLUMNS, row))) for row in rows]

async def ainsert_analyzed_mood(mood: AnalyzedMood) -> None:
    await asyncio.to_thread(insert_analyzed_mood, mood)

async def aquery_analyzed_comments(limit: int | None = None) -> list[AnalyzedMood]:
    return await asyncio.to_thread(query_analyzed_comments, limit)

def close_connection() -> None:
    """Flush and close the embedded DuckDB connection (call on shutdown)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None