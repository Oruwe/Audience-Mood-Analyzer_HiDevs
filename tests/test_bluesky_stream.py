"""Tests for ingestion.bluesky_stream's pure helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from ingestion.bluesky_stream import (
    JetstreamCommit,
    JetstreamMessage,
    JetstreamRecord,
    _build_url,
    _matches,
    _to_raw_comment,
)


def test_matches_is_case_insensitive_substring():
    assert _matches("Loving the new AI Feature", ["ai"]) is True
    assert _matches("nothing relevant here", ["ai", "tech"]) is False


def test_build_url_includes_langs_when_given():
    url = _build_url(["en"])
    assert "wantedCollections=app.bsky.feed.post" in url
    assert "wantedLangs=en" in url


def test_build_url_omits_langs_when_absent():
    url = _build_url(None)
    assert "wantedLangs" not in url


def test_to_raw_comment_uses_created_at_when_present():
    msg = JetstreamMessage(
        did="did:plc:abc123",
        time_us=1_700_000_000_000_000,
        kind="commit",
        commit=JetstreamCommit(
            operation="create", collection="app.bsky.feed.post", rkey="rk1",
            record=JetstreamRecord(text="hello ai", createdAt="2024-01-01T00:00:00Z"),
        ),
    )
    comment = _to_raw_comment(msg)

    assert comment.id == "bsky:did:plc:abc123:rk1"
    assert comment.platform == "bluesky"
    assert comment.author_id == "did:plc:abc123"
    assert comment.timestamp == datetime(2024, 1, 1, tzinfo=UTC)


def test_to_raw_comment_falls_back_to_time_us_when_created_at_missing():
    msg = JetstreamMessage(
        did="did:plc:xyz",
        time_us=1_700_000_000_000_000,
        kind="commit",
        commit=JetstreamCommit(
            operation="create", collection="app.bsky.feed.post", rkey="rk2",
            record=JetstreamRecord(text="hello tech"),
        ),
    )
    comment = _to_raw_comment(msg)

    assert comment.timestamp == datetime.fromtimestamp(
        1_700_000_000_000_000 / 1_000_000, tz=UTC
    )


def test_jetstream_models_ignore_unknown_fields():
    msg = JetstreamMessage.model_validate_json(
        '{"did": "did:plc:x", "time_us": 1, "kind": "identity", "surprise_field": 42}'
    )
    assert msg.kind == "identity"
    assert msg.commit is None
