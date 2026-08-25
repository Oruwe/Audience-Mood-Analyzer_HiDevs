"""Ingestion broker: single entry point over mock/live sources with dedup.

Wraps the raw async generators so downstream workers see each distinct
(platform, text) pair at most once — repeated pool draws in mock mode and
duplicate reposts on the firehose are dropped here.
"""

import logging
from collections.abc import AsyncIterator

from ingestion.bluesky_stream import generate_bluesky_stream
from ingestion.mock_stream import generate_mock_stream
from schemas import RawComment

logger = logging.getLogger(__name__)

_DUP_LOG_EVERY = 50  # emit one info line per N suppressed duplicates


def _dedup_key(comment: RawComment) -> str:
    return f"{comment.platform}::{comment.text.strip().lower()}"


async def stream_inbound_comments(
    source_mode: str = "mock",
    keywords: list[str] | None = None,
) -> AsyncIterator[RawComment]:
    """Yield comments from the selected source, suppressing exact duplicates."""
    if source_mode == "mock":
        source = generate_mock_stream()
    elif source_mode == "bluesky":
        source = generate_bluesky_stream(keywords=keywords)
    else:
        raise ValueError(
            f"Unknown source_mode: {source_mode!r} (expected 'mock' or 'bluesky')"
        )

    seen: set[str] = set()
    dropped = 0
    async for comment in source:
        key = _dedup_key(comment)
        if key in seen:
            dropped += 1
            if dropped % _DUP_LOG_EVERY == 0:
                logger.info("broker: suppressed %d duplicate comments so far", dropped)
            continue
        seen.add(key)
        yield comment
