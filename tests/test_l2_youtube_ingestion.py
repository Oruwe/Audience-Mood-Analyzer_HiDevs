"""L2 contract tests — ingestion.youtube against a stubbed YouTube Data API
(httpx.MockTransport, no real network, no API key needed).

Covers: pagination across playlistItems/videos/commentThreads, the quota
pre-flight estimate matching SPEC §4.4's formula, clean refusal before any
comment is pulled when the estimate exceeds budget, and that the pull
wires through ingestion.normalizer + ingestion.dedup as SPEC §5's
architecture diagram requires.
"""

import asyncio
from urllib.parse import parse_qs

import httpx
import pytest

from ingestion.dedup import ContentDeduplicator
from ingestion.youtube import (
    MAX_COMMENTS_PER_VIDEO_DEFAULT,
    QuotaExceededError,
    QuotaLedger,
    estimate_channel_analysis,
    fetch_channel_comments,
)

API_KEY = "test-key"


def _channel_response():
    return {
        "items": [{
            "id": "UCabc",
            "snippet": {"title": "Test Channel"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UUabc"}},
            "statistics": {"videoCount": "3"},
        }]
    }


def _playlist_items_response():
    return {"items": [{"contentDetails": {"videoId": f"v{i}"}} for i in (1, 2, 3)]}


def _videos_response():
    return {
        "items": [
            {"id": "v1", "snippet": {"title": "Video One"}, "statistics": {"commentCount": "2"}},
            {"id": "v2", "snippet": {"title": "Video Two"}, "statistics": {"commentCount": "150"}},
            {"id": "v3", "snippet": {"title": "Video Three"}, "statistics": {"commentCount": "0"}},
        ]
    }


def _thread_item(comment_id: str, text: str, author: str = "Alice") -> dict:
    return {
        "snippet": {
            "topLevelComment": {
                "id": comment_id,
                "snippet": {
                    "textOriginal": text,
                    "authorDisplayName": author,
                    "authorChannelId": {"value": f"UC_{author}"},
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "likeCount": 3,
                },
            }
        },
        "replies": {"comments": []},
    }


def _build_handler(calls: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        path = request.url.path
        params = parse_qs(request.url.query.decode())

        if path.endswith("/channels"):
            return httpx.Response(200, json=_channel_response())
        if path.endswith("/playlistItems"):
            return httpx.Response(200, json=_playlist_items_response())
        if path.endswith("/videos"):
            return httpx.Response(200, json=_videos_response())
        if path.endswith("/commentThreads"):
            video_id = params["videoId"][0]
            page_token = (params.get("pageToken") or [None])[0]
            # The real API never returns more than maxResults. Honouring it
            # here matters now that the caller relies on it to bound the
            # pull: a stub that ignored maxResults would let a broken cap
            # pass this suite.
            max_results = int((params.get("maxResults") or ["100"])[0])
            if video_id == "v1":
                return httpx.Response(200, json={
                    "items": [
                        _thread_item("c1", "  First   comment  "),
                        # same normalized text as c1 -> dedup must drop this one
                        _thread_item("c1-dup", "First comment"),
                    ][:max_results],
                })
            if video_id == "v2" and page_token is None:
                items = [_thread_item(f"c2-{i}", f"comment number {i}") for i in range(100)]
                items[0] = _thread_item("c2-email", "reach me at jane@example.com please")
                return httpx.Response(
                    200, json={"items": items[:max_results], "nextPageToken": "page2"}
                )
            if video_id == "v2" and page_token == "page2":
                items = [_thread_item(f"c2b-{i}", f"comment number b{i}") for i in range(50)]
                return httpx.Response(200, json={"items": items[:max_results]})
            raise AssertionError(f"unexpected commentThreads call: {request.url}")
        raise AssertionError(f"unexpected path: {path}")

    return handler


def _client(calls: list[str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_build_handler(calls)))


async def _collect(agen):
    return [item async for item in agen]


def test_estimate_matches_spec_quota_formula():
    calls: list[str] = []
    ledger = QuotaLedger(daily_budget=10_000)

    async def scenario():
        async with _client(calls) as client:
            return await estimate_channel_analysis(
                "@testchan", client=client, api_key=API_KEY, ledger=ledger
            )

    estimate = asyncio.run(scenario())

    assert estimate.channel.channel_id == "UCabc"
    assert estimate.channel.uploads_playlist_id == "UUabc"
    assert [v.video_id for v in estimate.videos] == ["v1", "v2", "v3"]
    assert estimate.total_comment_count == 2 + 150 + 0
    # 1 (channels.list) + 1 (playlistItems.list) + 1 (videos.list, batched)
    assert estimate.units_already_spent_on_estimate == 3
    # Per-video cap (70) < page size (100), so every video with any comments
    # costs exactly one page: v1 -> 1, v2 (150 comments, capped to 70) -> 1,
    # v3 (none) -> 0. Before the cap existed v2 alone billed 2 pages here.
    assert estimate.units_required_for_comment_pull == 2
    assert estimate.total_units_required == 5
    assert ledger.spent == 3  # only the cheap estimate has been charged so far


def test_comment_pull_refuses_cleanly_before_pulling_anything_when_over_budget():
    calls: list[str] = []
    # Enough budget for the 3-unit estimate, not enough for the 2-unit pull
    # that follows it. (The pull is 2 units rather than 3 because of the
    # per-video comment cap -- see test_estimate_matches_spec_quota_formula.)
    ledger = QuotaLedger(daily_budget=4)

    async def scenario():
        async with _client(calls) as client:
            estimate = await estimate_channel_analysis(
                "@testchan", client=client, api_key=API_KEY, ledger=ledger
            )
            spent_after_estimate = ledger.spent
            gen = fetch_channel_comments(estimate, client=client, api_key=API_KEY, ledger=ledger)
            with pytest.raises(QuotaExceededError):
                await _collect(gen)
            return spent_after_estimate

    spent_after_estimate = asyncio.run(scenario())

    assert spent_after_estimate == 3
    # Refused before spending anything on the pull -- never half-fail.
    assert ledger.spent == 3
    assert not any("commentThreads" in c for c in calls)


def test_full_pull_wires_normalizer_and_dedup():
    calls: list[str] = []
    ledger = QuotaLedger(daily_budget=10_000)
    dedup = ContentDeduplicator(max_size=1000, ttl_seconds=3600)

    async def scenario():
        async with _client(calls) as client:
            estimate = await estimate_channel_analysis(
                "@testchan", client=client, api_key=API_KEY, ledger=ledger
            )
            comments = await _collect(
                fetch_channel_comments(
                    estimate, client=client, api_key=API_KEY, ledger=ledger, dedup=dedup
                )
            )
            return comments

    comments = asyncio.run(scenario())

    # v3 has 0 comments per the pre-flight estimate -> never queried at all.
    assert not any("videoId=v3" in c for c in calls)

    # v1: 2 raw threads, same normalized text -> dedup drops the second.
    v1_comments = [c for c in comments if c.video_id == "v1"]
    assert len(v1_comments) == 1
    assert v1_comments[0].text == "First comment"  # normalizer collapsed the whitespace

    # v2 has 150 comments across two pages, but the per-video cap stops the
    # pull at MAX_COMMENTS_PER_VIDEO_DEFAULT. This is the bound the OOM
    # postmortem asked for: the corpus size is a property of our config, not
    # of how popular the video happens to be.
    v2_comments = [c for c in comments if c.video_id == "v2"]
    assert len(v2_comments) == MAX_COMMENTS_PER_VIDEO_DEFAULT

    # ...and the second page was never requested, so the cap saved quota
    # rather than merely discarding comments we had already paid for.
    assert not any("pageToken=page2" in c for c in calls)

    # "Top" comments, not "most recent": a truncated pull ordered by time
    # would be a different sample entirely, so the ordering is part of the
    # contract now that truncation is guaranteed.
    v2_calls = [c for c in calls if "videoId=v2" in c]
    assert v2_calls and all("order=relevance" in c for c in v2_calls)
    assert all(f"maxResults={MAX_COMMENTS_PER_VIDEO_DEFAULT}" in c for c in v2_calls)

    # normalizer.sanitize_comment_text ran on every comment, not just v1's.
    masked = next(c for c in v2_comments if "[EMAIL]" in c.text)
    assert "jane@example.com" not in masked.text

    assert all(c.platform == "youtube" for c in comments)
    assert len(comments) == 1 + MAX_COMMENTS_PER_VIDEO_DEFAULT

    # Estimate (3) + actual pull (1 page for v1, 1 capped page for v2, v3
    # skipped) -- and the ledger must agree with the pre-flight estimate,
    # which is the whole point of computing one.
    assert ledger.spent == 3 + 2
