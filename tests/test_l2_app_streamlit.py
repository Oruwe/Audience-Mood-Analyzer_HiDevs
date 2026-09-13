"""L2 smoke tests — app.py's actual Streamlit page, via Streamlit's own
`AppTest` harness (ships with the `streamlit` package already in
requirements.txt — no new dependency).

The page now also renders a "Run live accuracy benchmark" button (the
Metrics Usage panel, app.py's `_render_eval_section`) above the channel
input, so `at.button` has more than one entry — these tests select the
Analyze button by its explicit `key="analyze_button"` (`_analyze_button`
below) rather than by position, so they can't accidentally click the eval
button and fire a real OpenRouter call against a fake key.

Scope note: `AppTest.from_file` re-executes app.py in its own isolated
namespace rather than reusing an `import app` from this process, so
monkeypatching an externally-imported `app` module has no effect on what
AppTest runs (confirmed empirically — a patched `app.estimate_channel_analysis`
was ignored and the real function ran a real network call). These tests
are therefore scoped to what's genuinely reachable without a real YouTube/
OpenRouter key: the missing-config guard, input validation state, and a
malformed URL (which fails locally in ingestion.youtube.parse_youtube_url,
before any network call). The actual business logic these paths sit on
top of — preflight/launch_analysis/find_cached_analysis/load_progress/
load_insights — is unit-tested with real monkeypatching in
tests/test_l2_app_logic.py, where `import app` + monkeypatch does work.

A cold import of app.py (transitively importing litellm) takes several
seconds, well past AppTest's 3s default -- default_timeout=30 below is
generous for that, not a sign these tests are slow to *run*.
"""

import asyncio
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
_TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/audience_mood_analyzer"


def _clear_eval_runs() -> None:
    """The eval-benchmark panel's "last run" (storage.postgres.
    model_eval_runs) is deliberately global/persistent (it's meant to
    survive a real restart) -- which means a *real* failed run recorded
    against this same local dev Postgres (e.g. an earlier bug that let a
    test click that button for real, before `_analyze_button` existed)
    would otherwise leak into every one of these tests as an extra
    st.error(...) on the page forever. Clear the slate before each
    configured AppTest run so these tests see the same "not run yet" state
    a fresh deployment would.
    """
    async def _scenario() -> None:
        import asyncpg
        conn = await asyncpg.connect(_TEST_DATABASE_URL)
        try:
            await conn.execute("DELETE FROM model_eval_runs")
        except asyncpg.exceptions.UndefinedTableError:
            pass  # schema not created yet in this DB -- nothing to clear
        finally:
            await conn.close()

    try:
        asyncio.run(_scenario())
    except OSError:
        pass  # no local Postgres reachable -- these tests will fail for that reason anyway


def _app(monkeypatch, *, configured: bool) -> AppTest:
    if configured:
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-yt-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-or-key")
        monkeypatch.setenv("DATABASE_URL", _TEST_DATABASE_URL)
        _clear_eval_runs()
    else:
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


def _analyze_button(at: AppTest):
    return next(b for b in at.button if b.key == "analyze_button")


def test_missing_config_shows_error_and_stops_before_rendering_input(monkeypatch):
    at = _app(monkeypatch, configured=False)
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "YOUTUBE_API_KEY" in at.error[0].value
    assert "OPENROUTER_API_KEY" in at.error[0].value
    assert "DATABASE_URL" in at.error[0].value
    # st.stop() must have actually stopped the script -- no input widgets
    # ever get rendered when configuration is missing.
    assert len(at.text_input) == 0
    assert len(at.button) == 0


def test_page_renders_title_and_input_when_configured(monkeypatch):
    at = _app(monkeypatch, configured=True)
    at.run()

    assert not at.exception
    assert at.title[0].value == "🎥 Audience Mood Analyzer"
    assert len(at.text_input) == 1
    # The Analyze button, plus the eval section's "Run live accuracy
    # benchmark" button (_render_eval_section) -- see module docstring.
    assert len(at.button) == 2
    assert _analyze_button(at) is not None


def test_analyze_button_is_disabled_until_something_is_typed(monkeypatch):
    at = _app(monkeypatch, configured=True)
    at.run()
    assert _analyze_button(at).disabled is True

    at.text_input[0].set_value("https://www.youtube.com/@mkbhd")
    at.run()
    assert _analyze_button(at).disabled is False

    at.text_input[0].set_value("   ")  # whitespace-only -- still "empty"
    at.run()
    assert _analyze_button(at).disabled is True


def test_malformed_url_shows_a_clear_error_without_touching_the_network(monkeypatch):
    at = _app(monkeypatch, configured=True)
    at.run()
    at.text_input[0].set_value("not a youtube url at all")
    at.run()

    _analyze_button(at).click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "not a youtube url at all" in at.error[0].value
    # Still on the input screen -- no job was ever started.
    assert len(at.status) == 0
