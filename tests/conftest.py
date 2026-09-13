"""tests-level fixtures.

`pg_dsn` gates every Postgres integration test (tests/test_l2_postgres_*.py)
behind a real connectivity check, skipping cleanly rather than failing red
when no Postgres is reachable — e.g. a CI job with no Postgres service
configured yet (that lands in Phase 9's CI workflow), or a fresh sandbox
session where nobody has run `service postgresql start` yet. This sandbox
ships its own local Postgres 16 for development (SPEC §4.3: same schema,
same driver as production Neon/Supabase, just a different host).
"""

from __future__ import annotations

import asyncio
import os

import asyncpg
import pytest

DEFAULT_TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/audience_mood_analyzer"


def _pg_reachable(dsn: str) -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(dsn, timeout=2)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL", DEFAULT_TEST_DATABASE_URL)
    if not _pg_reachable(dsn):
        pytest.skip(f"Postgres not reachable at {dsn!r} -- skipping DB integration tests")
    return dsn
