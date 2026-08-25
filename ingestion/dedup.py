"""Phase-1 content deduplication for the ingestion layer.

Every inbound comment is reduced to a normalised SHA-256 fingerprint;
ContentDeduplicator remembers recently-seen fingerprints and reports whether
a fingerprint has already appeared inside its TTL window.

Backends:
  * In-memory LRU (default, zero-dependency): bounded OrderedDict with
    per-entry monotonic-clock expiry. This is the CONVENTIONS-compliant
    default — no external services required.
  * Redis (opt-in): atomic ``SET NX EX`` when ``REDIS_URL`` is present;
    imported lazily so the default path stays dependency-free.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

_KEY_PREFIX = "dedup:"


def fingerprint_text(text: str) -> str:
    """Case- and whitespace-insensitive 64-char SHA-256 hex fingerprint."""
    normalised = " ".join(text.casefold().split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class _MemoryBackend:
    """Bounded LRU map of fingerprint -> monotonic expiry timestamp."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max(1, max_size)
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, float] = OrderedDict()

    async def is_duplicate(self, fingerprint: str) -> bool:
        now = time.monotonic()
        expiry = self._entries.get(fingerprint)
        if expiry is not None:
            if expiry > now:
                self._entries.move_to_end(fingerprint)  # mark as recently used
                return True
            del self._entries[fingerprint]              # TTL elapsed
            return False
        self._entries[fingerprint] = now + self._ttl
        self._entries.move_to_end(fingerprint)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)           # evict least-recently-used
        return False

    async def aclose(self) -> None:
        self._entries.clear()


class _RedisBackend:
    """Opt-in Redis backend using atomic SET NX EX (requires REDIS_URL)."""

    def __init__(self, url: str, ttl_seconds: float) -> None:
        import redis.asyncio as aioredis  # lazy import: unused on default path

        # Redis EX granularity is whole seconds; clamp so SET never rejects ex=0.
        self._ttl = max(int(ttl_seconds), 1)
        self._redis = aioredis.from_url(url)

    async def is_duplicate(self, fingerprint: str) -> bool:
        created = await self._redis.set(
            f"{_KEY_PREFIX}{fingerprint}", "1", nx=True, ex=self._ttl
        )
        return not created  # NX set succeeded => first sighting

    async def aclose(self) -> None:
        await self._redis.aclose()


class ContentDeduplicator:
    """Async duplicate detector with pluggable Redis / in-memory backends."""

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 3600.0,
                 redis_url: str | None = None) -> None:
        url = redis_url or os.environ.get("REDIS_URL", "")
        self._backend: _MemoryBackend | _RedisBackend
        if url:
            try:
                self._backend = _RedisBackend(url, ttl_seconds)
                logger.info("Deduplicator backend: Redis (%s, ttl=%ss)", url, ttl_seconds)
                return
            except Exception as exc:  # noqa: BLE001 — degrade gracefully to memory
                logger.warning("Redis backend unavailable (%s); falling back to memory", exc)
        self._backend = _MemoryBackend(max_size=max_size, ttl_seconds=ttl_seconds)
        logger.info(
            "Deduplicator backend: in-memory LRU (max_size=%d, ttl_seconds=%s)",
            max_size, ttl_seconds,
        )

    async def is_duplicate(self, fingerprint: str) -> bool:
        """True when *fingerprint* was already recorded inside the TTL window."""
        return await self._backend.is_duplicate(fingerprint)

    async def aclose(self) -> None:
        """Release backend resources (safe to call more than once)."""
        await self._backend.aclose()
