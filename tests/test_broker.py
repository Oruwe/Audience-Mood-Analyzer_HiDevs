"""Tests for ingestion.broker.stream_inbound_comments."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ingestion.broker import stream_inbound_comments
from schemas import RawComment


def _comment(id_: str, text: str) -> RawComment:
    return RawComment(
        id=id_, platform="twitter", text=text, timestamp=datetime.now(UTC)
    )


async def test_unknown_source_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown source_mode"):
        async for _ in stream_inbound_comments(source_mode="carrier-pigeon"):
            pass


async def test_duplicate_comments_are_suppressed_after_sanitizing(monkeypatch):
    async def fake_stream():
        yield _comment("1", "Check this out!")
        yield _comment("2", "check   this out!")  # same after whitespace/case fold
        yield _comment("3", "a completely different comment")

    monkeypatch.setattr("ingestion.broker.generate_mock_stream", lambda: fake_stream())

    seen = [c.id async for c in stream_inbound_comments(source_mode="mock")]
    assert seen == ["1", "3"]


async def test_sanitization_runs_before_fingerprinting(monkeypatch):
    """Two URLs differing only by tracking params must collapse to one fingerprint."""

    async def fake_stream():
        yield _comment("1", "See https://example.com/x?utm_source=a")
        yield _comment("2", "See https://example.com/x?utm_source=b")

    monkeypatch.setattr("ingestion.broker.generate_mock_stream", lambda: fake_stream())

    seen = [c.id async for c in stream_inbound_comments(source_mode="mock")]
    assert seen == ["1"]
