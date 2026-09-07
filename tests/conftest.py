"""Shared pytest fixtures.

The suite is fully hermetic: no real network calls, no shared DuckDB file
between tests, and no reliance on real provider API keys. Every test that
needs a warehouse gets its own tmp_path file via the ``db_path`` fixture,
and every test that needs an LLM response stubs the provider call directly
rather than hitting Gemini/Groq/Langfuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Mirrors the sys.path bootstrap already used by run_test.py / evals/benchmark.py
# / scripts/*.py: the project keeps top-level modules (schemas.py, server.py,
# engine/, ingestion/, storage/) importable from the repo root rather than a
# src/ layout, so tests need the same explicit path insertion.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no provider/Redis/Langfuse keys unless it sets its own."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "REDIS_URL",
                "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point storage.db at an isolated, initially-nonexistent DuckDB file."""
    import storage.db as db_mod

    path = tmp_path / "test_analytics.duckdb"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    return path
