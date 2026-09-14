"""L1 unit tests — ingestion.dedup (fingerprinting + in-memory LRU/TTL).

Ported from the old run_test.py::verify_deduplicator (SPEC §2: "keep,
repurpose to kill bot spam and copypasta within a channel"). Uses
asyncio.run() directly rather than pytest-asyncio, which isn't a repo
dependency.
"""

import asyncio
import os

from ingestion.dedup import ContentDeduplicator, fingerprint_text


def _run(coro):
    return asyncio.run(coro)


def test_fingerprint_is_64_char_sha256_hex():
    fp = fingerprint_text("  This Product Is Amazing  ")
    assert len(fp) == 64
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_fingerprint_is_case_and_whitespace_insensitive():
    fp_a = fingerprint_text("  This Product Is Amazing  ")
    fp_b = fingerprint_text("this product is amazing")
    assert fp_a == fp_b


def test_first_and_second_sighting():
    async def scenario():
        saved = os.environ.pop("REDIS_URL", None)  # force deterministic memory backend
        try:
            dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)
            fp = fingerprint_text("hello world")
            first = await dedup.is_duplicate(fp)
            second = await dedup.is_duplicate(fp)
            await dedup.aclose()
            return first, second
        finally:
            if saved is not None:
                os.environ["REDIS_URL"] = saved

    first, second = _run(scenario())
    assert first is False
    assert second is True


def test_distinct_texts_each_flag_independently_on_repeat():
    async def scenario():
        dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)
        distinct = [fingerprint_text(f"genuinely different comment #{i}") for i in range(3)]
        assert len(set(distinct)) == 3
        for fp in distinct:
            await dedup.is_duplicate(fp)
        hits = [await dedup.is_duplicate(fp) for fp in distinct]
        await dedup.aclose()
        return hits

    assert all(_run(scenario()))


def test_lru_evicts_oldest_beyond_max_size():
    async def scenario():
        small = ContentDeduplicator(max_size=3, ttl_seconds=3600)
        fps = [fingerprint_text(f"evict-me-{i}") for i in range(4)]
        for fp in fps[:3]:
            await small.is_duplicate(fp)
        await small.is_duplicate(fps[3])  # 4th insert evicts fps[0]
        evicted_seen_fresh = await small.is_duplicate(fps[0])
        recent_still_known = await small.is_duplicate(fps[3])
        await small.aclose()
        return evicted_seen_fresh, recent_still_known

    evicted_seen_fresh, recent_still_known = _run(scenario())
    assert evicted_seen_fresh is False  # evicted -> treated as first sighting again
    assert recent_still_known is True


def test_ttl_expiry_treats_stale_entries_as_fresh():
    async def scenario():
        flash = ContentDeduplicator(max_size=10, ttl_seconds=0)  # instantly stale
        fp = fingerprint_text("ttl probe")
        await flash.is_duplicate(fp)
        second = await flash.is_duplicate(fp)
        await flash.aclose()
        return second

    assert _run(scenario()) is False
