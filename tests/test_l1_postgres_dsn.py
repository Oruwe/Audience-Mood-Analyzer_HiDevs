"""L1 unit tests — storage.postgres's SSL-mode DSN handling.

Found while shipping to Render: its own query tool hit "SSL/TLS required"
going through the external network path to a fresh Postgres instance,
while the app itself connects over Render's internal network for the same
DATABASE_URL — different paths, possibly different SSL requirements.
`sslmode=prefer` makes one connection string correct either way (and
against this sandbox's local dev Postgres, and Neon/Supabase, too).
"""

import pytest

from storage.postgres import _with_default_sslmode


def test_adds_sslmode_prefer_when_absent():
    result = _with_default_sslmode("postgresql://user:pass@host/db")
    assert "sslmode=prefer" in result
    assert result.startswith("postgresql://user:pass@host/db?")


def test_leaves_an_explicit_sslmode_untouched():
    dsn = "postgresql://user:pass@host/db?sslmode=require"
    assert _with_default_sslmode(dsn) == dsn


@pytest.mark.parametrize("existing_mode", ["disable", "allow", "require", "verify-full"])
def test_never_overrides_any_explicit_choice(existing_mode):
    dsn = f"postgresql://user:pass@host/db?sslmode={existing_mode}"
    assert _with_default_sslmode(dsn) == dsn


def test_preserves_other_query_params_when_adding_sslmode():
    result = _with_default_sslmode("postgresql://user:pass@host/db?application_name=app")
    assert "application_name=app" in result
    assert "sslmode=prefer" in result


def test_result_is_still_a_connectable_dsn_shape():
    result = _with_default_sslmode("postgresql://user:pass@host:5432/db")
    assert result.startswith("postgresql://user:pass@host:5432/db?sslmode=prefer")
