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
