"""L2 smoke tests — app.py's actual Streamlit page, via Streamlit's own
`AppTest` harness (ships with the `streamlit` package already in
requirements.txt — no new dependency).

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

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _app(monkeypatch, *, configured: bool) -> AppTest:
    if configured:
        monkeypatch.setenv("YOUTUBE_API_KEY", "fake-yt-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-or-key")
        monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/audience_mood_analyzer")
    else:
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
    return AppTest.from_file(str(APP_PATH), default_timeout=30)


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
    assert len(at.button) == 1


def test_analyze_button_is_disabled_until_something_is_typed(monkeypatch):
    at = _app(monkeypatch, configured=True)
    at.run()
    assert at.button[0].disabled is True

    at.text_input[0].set_value("https://www.youtube.com/@mkbhd")
    at.run()
    assert at.button[0].disabled is False

    at.text_input[0].set_value("   ")  # whitespace-only -- still "empty"
    at.run()
    assert at.button[0].disabled is True


def test_malformed_url_shows_a_clear_error_without_touching_the_network(monkeypatch):
    at = _app(monkeypatch, configured=True)
    at.run()
    at.text_input[0].set_value("not a youtube url at all")
    at.run()

    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "not a youtube url at all" in at.error[0].value
    # Still on the input screen -- no job was ever started.
    assert len(at.status) == 0
