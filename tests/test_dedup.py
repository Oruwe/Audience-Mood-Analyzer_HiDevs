"""Tests for ingestion.dedup: fingerprinting and the in-memory LRU backend."""

from __future__ import annotations

from ingestion.dedup import ContentDeduplicator, fingerprint_text


def test_fingerprint_is_64_char_sha256_hex():
    fp = fingerprint_text("This Product Is Amazing")
    assert len(fp) == 64
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_fingerprint_is_case_and_whitespace_insensitive():
    fp_a = fingerprint_text("  This Product Is Amazing  ")
    fp_b = fingerprint_text("this product is amazing")
    assert fp_a == fp_b


def test_distinct_texts_get_distinct_fingerprints():
    fps = {fingerprint_text(f"genuinely different comment #{i}") for i in range(3)}
    assert len(fps) == 3


async def test_first_sighting_not_duplicate_second_sighting_is():
    dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)
    fp = fingerprint_text("hello")
    assert await dedup.is_duplicate(fp) is False
    assert await dedup.is_duplicate(fp) is True
    await dedup.aclose()


async def test_text_variant_resolves_to_same_fingerprint_and_flags_duplicate():
    dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)
    fp_a = fingerprint_text("  This Product Is Amazing  ")
    fp_b = fingerprint_text("this product is amazing")
    await dedup.is_duplicate(fp_a)
    assert await dedup.is_duplicate(fp_b) is True
    await dedup.aclose()


async def test_independent_texts_each_flag_on_repeat():
    dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)
    fps = [fingerprint_text(f"independent comment #{i}") for i in range(3)]
    for fp in fps:
        await dedup.is_duplicate(fp)
    assert all([await dedup.is_duplicate(fp) for fp in fps])
    await dedup.aclose()


async def test_lru_evicts_oldest_beyond_max_size():
    dedup = ContentDeduplicator(max_size=3, ttl_seconds=3600)
    fps = [fingerprint_text(f"evict-me-{i}") for i in range(4)]
    for fp in fps[:3]:
        await dedup.is_duplicate(fp)
    await dedup.is_duplicate(fps[3])  # 4th insert evicts fps[0]

    assert await dedup.is_duplicate(fps[0]) is False  # forgotten -> "new" sighting
    assert await dedup.is_duplicate(fps[3]) is True   # still remembered
    await dedup.aclose()


async def test_ttl_expiry_treats_stale_entries_as_fresh():
    dedup = ContentDeduplicator(max_size=10, ttl_seconds=0)
    fp = fingerprint_text("ttl probe")
    await dedup.is_duplicate(fp)
    assert await dedup.is_duplicate(fp) is False
    await dedup.aclose()


async def test_dedup_falls_back_to_memory_when_redis_is_unavailable():
    """No real Redis server exists in the test sandbox (and the ``redis``
    package isn't even installed, since it's an opt-in dependency) — the
    constructor must degrade to the in-memory backend rather than raising.
    """
    dedup = ContentDeduplicator(redis_url="redis://localhost:1")
    fp = fingerprint_text("still works without redis")
    assert await dedup.is_duplicate(fp) is False
    assert await dedup.is_duplicate(fp) is True
    await dedup.aclose()
