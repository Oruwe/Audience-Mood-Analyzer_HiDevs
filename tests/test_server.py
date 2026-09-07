"""Tests for the FastAPI facade in server.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

import server as server_mod
from schemas import EnrichedCommentRecord, PrimaryIntent, RecommendedAction, Sentiment

client = TestClient(server_mod.app)


def _record() -> EnrichedCommentRecord:
    return EnrichedCommentRecord(
        comment_id="c1", platform="twitter", author_handle="@u", raw_text="hi",
        sentiment=Sentiment.POSITIVE, confidence=0.9,
        primary_intent=PrimaryIntent.PRAISE_ENDORSEMENT, urgency_score=0.1,
        emotional_drivers=[], summary="nice", recommended_action=RecommendedAction.IGNORE,
        suggested_reply_draft=None, brand_safety_flag=False, embedding=None,
        cluster_id=None, latency_ms=5.0, model_used="test",
        processed_at=datetime.now(UTC),
    )


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_comments_returns_records(monkeypatch):
    monkeypatch.setattr(server_mod, "query_enriched_records", lambda limit: [_record()])
    resp = client.get("/comments", params={"limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["comment_id"] == "c1"


def test_comments_rejects_limit_out_of_bounds():
    resp = client.get("/comments", params={"limit": 0})
    assert resp.status_code == 422


def test_comments_returns_503_when_warehouse_missing(monkeypatch):
    def _raise(limit):
        raise FileNotFoundError()

    monkeypatch.setattr(server_mod, "query_enriched_records", _raise)
    resp = client.get("/comments")
    assert resp.status_code == 503


def test_metrics_returns_empty_dict_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(server_mod, "METRICS_PATH", tmp_path / "missing.json")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json() == {}


def test_metrics_returns_file_contents_when_present(monkeypatch, tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"accuracy": 0.87}), encoding="utf-8")
    monkeypatch.setattr(server_mod, "METRICS_PATH", path)
    resp = client.get("/metrics")
    assert resp.json() == {"accuracy": 0.87}
