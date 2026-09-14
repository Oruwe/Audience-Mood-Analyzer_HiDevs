"""L1 unit tests — ingestion.youtube.parse_youtube_url (no network)."""

import pytest

from ingestion.youtube import ParsedInput, UnparsableURLError, parse_youtube_url


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s",
    "https://youtube.com/watch?t=30s&v=dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=10",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
])
def test_video_urls_resolve_to_video_id(url):
    parsed = parse_youtube_url(url)
    assert parsed == ParsedInput(kind="video", video_id="dQw4w9WgXcQ")


def test_channel_id_url():
    parsed = parse_youtube_url("https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw")
    assert parsed == ParsedInput(kind="channel_id", channel_id="UCuAXFkgsw1L7xaCfnd5JJOw")


def test_bare_channel_id():
    parsed = parse_youtube_url("UCuAXFkgsw1L7xaCfnd5JJOw")
    assert parsed == ParsedInput(kind="channel_id", channel_id="UCuAXFkgsw1L7xaCfnd5JJOw")


def test_handle_url():
    parsed = parse_youtube_url("https://www.youtube.com/@mkbhd")
    assert parsed == ParsedInput(kind="handle", handle="@mkbhd")


def test_bare_handle():
    parsed = parse_youtube_url("@mkbhd")
    assert parsed == ParsedInput(kind="handle", handle="@mkbhd")


def test_legacy_username_url():
    parsed = parse_youtube_url("https://www.youtube.com/user/someuser")
    assert parsed == ParsedInput(kind="legacy_username", legacy_username="someuser")


def test_legacy_custom_url_explains_why_it_cannot_be_resolved():
    with pytest.raises(UnparsableURLError, match="search.list"):
        parse_youtube_url("https://www.youtube.com/c/SomeCustomName")


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "https://example.com/not-youtube",
    "https://www.youtube.com/watch",  # no v= param
    "not a url at all",
])
def test_unparsable_inputs_raise(bad):
    with pytest.raises(UnparsableURLError):
        parse_youtube_url(bad)
