"""Tests for engine.embedder.generate_embedding."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import engine.embedder as embedder_mod
from engine.embedder import EMBED_DIM, generate_embedding


async def test_no_api_key_uses_hash_fallback():
    vec = await generate_embedding("hello world")
    assert len(vec) == EMBED_DIM
    norm = math.sqrt(sum(v * v for v in vec))
    assert norm == pytest.approx(1.0)


async def test_hash_fallback_is_deterministic():
    a = await generate_embedding("the quick brown fox")
    b = await generate_embedding("the quick brown fox")
    assert a == b


async def test_hash_fallback_differs_for_different_text():
    a = await generate_embedding("alpha")
    b = await generate_embedding("beta")
    assert a != b


async def test_empty_text_returns_zero_vector():
    vec = await generate_embedding("")
    assert vec == [0.0] * EMBED_DIM


async def test_gemini_success_path_used_when_key_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    async def fake_aembedding(**kwargs):
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]}])

    monkeypatch.setattr(embedder_mod, "aembedding", fake_aembedding)
    vec = await generate_embedding("anything")

    assert len(vec) == EMBED_DIM
    assert vec[:3] == pytest.approx([0.1, 0.2, 0.3])
    assert all(v == 0.0 for v in vec[3:])


async def test_gemini_failure_falls_back_to_hash(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    async def fake_aembedding(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(embedder_mod, "aembedding", fake_aembedding)
    vec = await generate_embedding("hello world")

    assert vec == embedder_mod._hash_embedding("hello world")
