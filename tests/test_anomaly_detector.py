"""Tests for engine.anomaly_detector.detect_anomalies.

GEMINI_API_KEY is deliberately left unset in every test here (the autouse
_clean_provider_env fixture guarantees that): extract_trending_theme then
resolves to its statistical fallback with zero LLM calls, keeping this
whole module's tests hermetic without needing to stub litellm as well.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from engine.anomaly_detector import detect_anomalies
from schemas import EnrichedCommentRecord, PrimaryIntent, RecommendedAction, Sentiment
from storage.db import _ensure_schema, ainsert_enriched_record


def _record(i: int, urgency: float) -> EnrichedCommentRecord:
    return EnrichedCommentRecord(
        comment_id=f"radar-{i}", platform="bluesky", author_handle=f"@u{i}",
        raw_text=f"comment {i}", sentiment=Sentiment.NEGATIVE, confidence=0.9,
        primary_intent=PrimaryIntent.BUG_REPORT, urgency_score=urgency,
        emotional_drivers=["frustration"], summary=f"note {i}",
        recommended_action=RecommendedAction.ESCALATE_TO_SUPPORT,
        suggested_reply_draft=None, brand_safety_flag=False,
        embedding=None, cluster_id=None, latency_ms=10.0,
        model_used="hermetic", processed_at=datetime.now(UTC),
    )


async def test_no_warehouse_file_yet_returns_none(db_path):
    assert await detect_anomalies() is None


async def test_empty_but_initialised_warehouse_returns_none(db_path):
    with duckdb.connect(str(db_path)) as conn:
        _ensure_schema(conn)
    assert await detect_anomalies() is None


async def test_calm_window_does_not_alert(db_path):
    for i in range(4):
        await ainsert_enriched_record(_record(i, urgency=0.2))
    assert await detect_anomalies() is None


async def test_mean_trigger_raises_critical_alert(db_path):
    for i in range(4):
        await ainsert_enriched_record(_record(i, urgency=0.95))

    alert = await detect_anomalies()

    assert alert is not None
    assert alert.severity == "CRITICAL"
    assert len(alert.affected_comment_ids) == 4
    assert "mean urgency" in alert.trigger_reason


async def test_spike_only_window_raises_warning_alert(db_path):
    # 0.05 (not 0.10) keeps the float mean at 0.6875, safely under the
    # strict ">0.70" mean trigger so only the spike-count path fires.
    for i, urgency in enumerate([0.90, 0.90, 0.90, 0.05]):
        await ainsert_enriched_record(_record(100 + i, urgency))

    alert = await detect_anomalies()

    assert alert is not None
    assert alert.severity == "WARNING"
