"""Async text embedder: Gemini free-tier first, local hash fallback."""

import hashlib
import logging
import math
import os

from litellm import aembedding

logger = logging.getLogger(__name__)

EMBED_DIM = 768  # matches gemini/text-embedding-004 output


def _hash_embedding(text: str) -> list[float]:
    """Deterministic offline fallback: signed feature-hashing of tokens."""
    vec = [0.0] * EMBED_DIM
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % EMBED_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def generate_embedding(text: str) -> list[float]:
    """Return a fixed-dimension embedding vector for `text`."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            response = await aembedding(
                model="gemini/text-embedding-004",
                input=[text],
                api_key=api_key,
                timeout=30,
            )
            vec = list(response.data[0]["embedding"])
            return (vec + [0.0] * EMBED_DIM)[:EMBED_DIM]
        except Exception as exc:  # noqa: BLE001 — never fail ingestion on embedder
            logger.warning("Gemini embedding failed (%s); using local fallback", exc)
    return _hash_embedding(text)
