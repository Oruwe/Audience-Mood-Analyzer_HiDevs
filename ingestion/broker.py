"""Ingestion broker: single entry point over mock/live sources with dedup.

Wraps the raw async generators so downstream workers see each distinct
(platform, sanitised-text) pair at most once — repeated pool draws in mock
mode and duplicate reposts on the firehose are dropped here. Text passes
through the Phase-1 normaliser BEFORE fingerprinting, so tracking-param URL
variants collapse onto one fingerprint and PII never reaches the LLM or DuckDB.
"""

import logging
from collections.abc import AsyncIterator

from ingestion.bluesky_stream import generate_bluesky_stream
from ingestion.dedup import ContentDeduplicator, fingerprint_text
from ingestion.mock_stream import generate_mock_stream
from ingestion.normalizer import sanitize_comment_text
from schemas import RawComment

logger = logging.getLogger(__name__)

_DUP_LOG_EVERY = 50  # emit one info line per N suppressed duplicates


async def stream_inbound_comments(
    source_mode: str = "mock",
    keywords: list[str] | None = None,
) -> AsyncIterator[RawComment]:
    """Yield comments from the selected source, sanitised and de-duplicated."""
    if source_mode == "mock":
        source = generate_mock_stream()
    elif source_mode == "bluesky":
        source = generate_bluesky_stream(keywords=keywords)
    else:
        raise ValueError(
            f"Unknown source_mode: {source_mode!r} (expected 'mock' or 'bluesky')"
        )

    dedup = ContentDeduplicator()
    dropped = 0
    try:
        async for comment in source:
            clean_text = sanitize_comment_text(comment.text)
            fp = fingerprint_text(f"{comment.platform}::{clean_text}")
            if await dedup.is_duplicate(fp):
                dropped += 1
                if dropped % _DUP_LOG_EVERY == 0:
                    logger.info("broker: suppressed %d duplicate comments so far", dropped)
                continue
            yield comment.model_copy(update={"text": clean_text})
    finally:
        await dedup.aclose()
