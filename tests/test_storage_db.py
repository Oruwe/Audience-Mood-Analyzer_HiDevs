"""Tests for storage.db: insert/query round-trips and schema migration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

import storage.db as db
from schemas import EnrichedCommentRecord, PrimaryIntent, RecommendedAction, Sentiment


def _record(comment_id: str, **overrides) -> EnrichedCommentRecord:
    payload = dict(
        comment_id=comment_id, platform="reddit", author_handle="u/tester",
        raw_text="raw text", sentiment=Sentiment.NEGATIVE, confidence=0.8,
        primary_intent=PrimaryIntent.BUG_REPORT, urgency_score=0.5,
        emotional_drivers=["annoyance"], summary="summary text",
        recommended_action=RecommendedAction.COMMUNITY_REPLY,
        suggested_reply_draft=None, brand_safety_flag=False,
        embedding=[0.1, 0.2, 0.3], cluster_id=1, latency_ms=42.0,
        model_used="test-model", processed_at=datetime.now(UTC),
    )
    payload.update(overrides)
    return EnrichedCommentRecord(**payload)


def test_insert_and_query_round_trip(db_path):
    db.insert_enriched_record(_record("rt-1"))

    results = db.query_enriched_records()

    assert len(results) == 1
    got = results[0]
    assert got.comment_id == "rt-1"
    assert got.emotional_drivers == ["annoyance"]
    assert got.embedding == pytest.approx([0.1, 0.2, 0.3])
    assert got.sentiment == Sentiment.NEGATIVE


def test_insert_or_replace_upserts_on_comment_id(db_path):
    db.insert_enriched_record(_record("dup", summary="first"))
    db.insert_enriched_record(_record("dup", summary="second"))

    results = db.query_enriched_records()

    assert len(results) == 1
    assert results[0].summary == "second"


def test_query_respects_limit_and_orders_newest_first(db_path):
    now = datetime.now(UTC)
    db.insert_enriched_record(_record("old", processed_at=now - timedelta(minutes=10)))
    db.insert_enriched_record(_record("new", processed_at=now))

    results = db.query_enriched_records(limit=1)

    assert len(results) == 1
    assert results[0].comment_id == "new"


def test_get_comments_since_filters_by_window(db_path):
    now = datetime.now(UTC)
    db.insert_enriched_record(_record("recent", processed_at=now))
    db.insert_enriched_record(_record("stale", processed_at=now - timedelta(hours=1)))

    results = db.get_comments_since(minutes=15)

    assert [r.comment_id for r in results] == ["recent"]


def test_get_top_urgent_escalations_filters_by_threshold_or_action(db_path):
    db.insert_enriched_record(_record(
        "high-urgency", urgency_score=0.9, recommended_action=RecommendedAction.IGNORE,
    ))
    db.insert_enriched_record(_record(
        "escalation-flagged", urgency_score=0.1,
        recommended_action=RecommendedAction.ESCALATE_TO_PR,
    ))
    db.insert_enriched_record(_record(
        "routine", urgency_score=0.2, recommended_action=RecommendedAction.IGNORE,
    ))

    results = db.get_top_urgent_escalations(limit=5)

    assert {r.comment_id for r in results} == {"high-urgency", "escalation-flagged"}


def test_query_before_warehouse_exists_raises_file_not_found(db_path):
    with pytest.raises(FileNotFoundError):
        db.query_enriched_records()


def test_close_connection_is_a_safe_noop():
    db.close_connection()  # kept only for call-site compatibility; must not raise


# ---------------------------------------------------------------------------
# Schema migration: a pre-embedding warehouse must upgrade in place.
# ---------------------------------------------------------------------------

_LEGACY_TYPES = {
    "comment_id": "VARCHAR PRIMARY KEY",
    "platform": "VARCHAR",
    "author_handle": "VARCHAR",
    "raw_text": "VARCHAR",
    "sentiment": "VARCHAR",
    "confidence": "DOUBLE",
    "primary_intent": "VARCHAR",
    "urgency_score": "DOUBLE",
    "emotional_drivers": "VARCHAR",
    "summary": "VARCHAR",
    "recommended_action": "VARCHAR",
    "suggested_reply_draft": "VARCHAR",
    "brand_safety_flag": "BOOLEAN",
    "latency_ms": "DOUBLE",
    "processed_at": "TIMESTAMPTZ",
}

_LEGACY_VALUES = {
    "comment_id": "legacy-1",
    "platform": "twitter",
    "author_handle": "@legacy",
    "raw_text": "legacy raw text",
    "sentiment": "negative",
    "confidence": 0.7,
    "urgency_score": 0.6,
    "primary_intent": "bug_report",
    "emotional_drivers": '["frustration"]',
    "summary": "legacy summary",
    "recommended_action": "community_reply",
    "suggested_reply_draft": None,
    "brand_safety_flag": False,
    "latency_ms": 12.5,
}


def test_schema_migration_backfills_legacy_rows_and_archives_old_table(db_path):
    """Simulates the exact 'stale pre-embedding layout' _ensure_schema documents:
    embedding/cluster_id/model_used didn't exist yet in an older deployment.
    """
    legacy_columns = [c for c in db._COLUMNS if c not in ("embedding", "cluster_id", "model_used")]
    with duckdb.connect(str(db_path)) as conn:
        cols_sql = ", ".join(f'"{c}" {_LEGACY_TYPES[c]}' for c in legacy_columns)
        conn.execute(f"CREATE TABLE analyzed_comments ({cols_sql})")
        placeholders = ", ".join(["?"] * len(legacy_columns))
        values = [
            _LEGACY_VALUES[c] if c != "processed_at" else datetime.now(UTC)
            for c in legacy_columns
        ]
        conn.execute(f"INSERT INTO analyzed_comments VALUES ({placeholders})", values)

    results = db.query_enriched_records()

    assert len(results) == 1
    assert results[0].comment_id == "legacy-1"
    assert results[0].embedding is None
    assert results[0].cluster_id is None
    assert results[0].model_used is None

    with duckdb.connect(str(db_path), read_only=True) as conn:
        archived = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'analyzed_comments_legacy_%'"
        ).fetchall()
    assert len(archived) == 1


def test_parse_jsonish_handles_json_python_literal_and_garbage():
    assert db._parse_jsonish(None, "fallback") is None
    assert db._parse_jsonish(["already", "a", "list"], []) == ["already", "a", "list"]
    assert db._parse_jsonish('["a", "b"]', []) == ["a", "b"]
    assert db._parse_jsonish("['a', 'b']", []) == ["a", "b"]  # python-literal fallback path
    assert db._parse_jsonish("not json or python", "fallback") == "fallback"
