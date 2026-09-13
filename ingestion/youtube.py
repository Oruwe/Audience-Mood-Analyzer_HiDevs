"""SPEC §1 + §4.4 — YouTube ingestion: channel/video URL -> video list ->
comment pagination, with quota accounting and a clean refusal over budget.

Architecture (SPEC §5): this module is the only thing that talks to the
YouTube Data API. Everything it yields has already been through
ingestion.normalizer (PII masking, UTM stripping) and ingestion.dedup
(fingerprint + TTL) before the caller ever sees it.

Quota model (SPEC §4.4, numbers recorded in SPEC.md §4.4): every list call
this module uses (`channels.list`, `playlistItems.list`, `videos.list`,
`commentThreads.list`) costs a flat 1 unit regardless of `part` or
`maxResults`. The expensive `search.list` (100 units) is deliberately never
used — see `_UNRESOLVABLE_CUSTOM_URL_HINT` below for what that costs a user
who pastes a legacy `/c/CustomName` URL.

No retries here yet. SPEC §8 (Orchestration, a later phase) adds
`tenacity`-based exponential backoff around the YouTube and OpenRouter
calls; adding that dependency now would be jumping ahead of its phase.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

import httpx

from ingestion.dedup import ContentDeduplicator, fingerprint_text
from ingestion.normalizer import sanitize_comment_text
from schemas import RawComment

logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# --- quota costs (SPEC.md §4.4) --------------------------------------------
COST_CHANNELS_LIST = 1
COST_PLAYLIST_ITEMS_LIST = 1
COST_VIDEOS_LIST = 1
COST_COMMENT_THREADS_LIST = 1
COST_SEARCH_LIST = 100  # never spent by this module — recorded for the error message

PLAYLIST_ITEMS_PAGE_SIZE = 50   # playlistItems.list maxResults ceiling
VIDEOS_BATCH_SIZE = 50          # videos.list accepts up to 50 ids per call
COMMENT_THREADS_PAGE_SIZE = 100  # commentThreads.list maxResults ceiling

DAILY_QUOTA_BUDGET_DEFAULT = 10_000  # Google's default per-project daily allocation

# A full-channel analysis is capped to the N most recent uploads by default.
# Not in SPEC.md explicitly; added because "which video landed badly" (§3
# Block 3) is inherently a recency-weighted question, and an unbounded
# backfill against a 100k-sub channel's entire upload history would make
# both the quota estimate and the Stage A/B/C pipeline unpredictably large.
# One line to change; SPEC's own §11.1 config-boundary spirit applies here.
DEFAULT_MAX_VIDEOS = 50

_PACIFIC = ZoneInfo("America/Los_Angeles")  # YouTube quota resets midnight PT


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class YouTubeIngestionError(RuntimeError):
    """Base class for every error this module raises on purpose."""


class UnparsableURLError(YouTubeIngestionError):
    """The given string isn't a YouTube channel/video URL this module handles."""


class ChannelNotFoundError(YouTubeIngestionError):
    """The API returned zero channels for a resolvable reference."""


class YouTubeAPIError(YouTubeIngestionError):
    """The API returned a non-2xx response."""

    def __init__(self, path: str, status_code: int, body: str) -> None:
        self.path = path
        self.status_code = status_code
        self.body = body
        super().__init__(f"{path} -> HTTP {status_code}: {body[:500]}")


class QuotaExceededError(YouTubeIngestionError):
    """Refusing to proceed: the planned work would exceed the daily budget.

    SPEC §4.4: "If a channel analysis would exceed quota, say so in the UI
    up front. A clear refusal reads as competence; a spinner that dies reads
    as broken." Raised *before* any comment-pulling starts — never partway
    through (SPEC: "refuse cleanly ... never half-fail").
    """

    def __init__(self, required_units: int, remaining_units: int) -> None:
        self.required_units = required_units
        self.remaining_units = remaining_units
        super().__init__(
            f"This analysis needs an estimated {required_units} YouTube API "
            f"quota units, but only {remaining_units} remain today "
            "(resets at midnight Pacific Time)."
        )


_UNRESOLVABLE_CUSTOM_URL_HINT = (
    "Legacy /c/CustomName channel URLs can't be resolved with a 1-unit "
    f"lookup — only a search.list call ({COST_SEARCH_LIST} quota units, "
    "100x the cost of everything else this product does) can do that, and "
    "this product deliberately never spends quota there. Use the channel's "
    "@handle or its /channel/UC... URL instead."
)


# ---------------------------------------------------------------------------
# URL / handle parsing (no network — never charges quota)
# ---------------------------------------------------------------------------

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")


@dataclass(frozen=True)
class ParsedInput:
    kind: Literal["video", "channel_id", "handle", "legacy_username"]
    video_id: str | None = None
    channel_id: str | None = None
    handle: str | None = None
    legacy_username: str | None = None


def parse_youtube_url(raw: str) -> ParsedInput:
    """Classify *raw* as a video reference or a channel reference.

    Accepts a bare channel ID, a bare @handle, or a full URL in any of:
    /watch?v=, youtu.be/, /shorts/, /channel/UC..., /@handle, /user/name.
    Raises UnparsableURLError (including for legacy /c/ URLs, which are
    explained rather than silently misparsed) on anything else.
    """
    text = raw.strip()
    if not text:
        raise UnparsableURLError("Empty input.")

    if _CHANNEL_ID_RE.match(text):
        return ParsedInput(kind="channel_id", channel_id=text)
    if text.startswith("@") and "/" not in text:
        return ParsedInput(kind="handle", handle=text)

    candidate = text if "//" in text else f"https://{text}"
    parsed = urllib.parse.urlsplit(candidate)
    host = parsed.netloc.lower()
    for prefix in ("www.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        if _VIDEO_ID_RE.match(video_id):
            return ParsedInput(kind="video", video_id=video_id)
        raise UnparsableURLError(f"Could not extract a video ID from {raw!r}")

    if host != "youtube.com":
        raise UnparsableURLError(f"Not a recognised YouTube URL: {raw!r}")

    parts = [p for p in parsed.path.split("/") if p]
    query = urllib.parse.parse_qs(parsed.query)

    if parts and parts[0] == "watch":
        values = query.get("v")
        if values and _VIDEO_ID_RE.match(values[0]):
            return ParsedInput(kind="video", video_id=values[0])
        raise UnparsableURLError(f"Could not extract a video ID from {raw!r}")

    if parts and parts[0] == "shorts" and len(parts) > 1 and _VIDEO_ID_RE.match(parts[1]):
        return ParsedInput(kind="video", video_id=parts[1])

    if parts and parts[0] == "channel" and len(parts) > 1:
        return ParsedInput(kind="channel_id", channel_id=parts[1])

    if parts and parts[0] == "user" and len(parts) > 1:
        return ParsedInput(kind="legacy_username", legacy_username=parts[1])

    if parts and parts[0] == "c":
        raise UnparsableURLError(_UNRESOLVABLE_CUSTOM_URL_HINT)

    if parts and parts[0].startswith("@"):
        return ParsedInput(kind="handle", handle=parts[0])

    raise UnparsableURLError(f"Unrecognised YouTube URL/handle format: {raw!r}")


# ---------------------------------------------------------------------------
# Quota ledger (SPEC §9: "Quota / rate counters: In-memory v1 -> Redis at
# multi-process") — one process, one ledger, resets on the Pacific-time day
# rolling over.
# ---------------------------------------------------------------------------

def _current_quota_day() -> str:
    return datetime.now(_PACIFIC).strftime("%Y-%m-%d")


@dataclass
class QuotaLedger:
    daily_budget: int = DAILY_QUOTA_BUDGET_DEFAULT
    _day: str = field(default_factory=_current_quota_day)
    _spent: int = 0

    def _roll_if_new_day(self) -> None:
        today = _current_quota_day()
        if today != self._day:
            self._day = today
            self._spent = 0

    @property
    def spent(self) -> int:
        self._roll_if_new_day()
        return self._spent

    @property
    def remaining(self) -> int:
        return self.daily_budget - self.spent

    def can_afford(self, units: int) -> bool:
        return units <= self.remaining

    def charge(self, units: int) -> None:
        self._roll_if_new_day()
        self._spent += units

    def require(self, units: int) -> None:
        """Raise QuotaExceededError instead of charging, if unaffordable."""
        if not self.can_afford(units):
            raise QuotaExceededError(required_units=units, remaining_units=self.remaining)


# ---------------------------------------------------------------------------
# Low-level API access
# ---------------------------------------------------------------------------

async def _get(client: httpx.AsyncClient, path: str, params: dict, api_key: str) -> dict:
    response = await client.get(
        f"{YOUTUBE_API_BASE}/{path}", params={**params, "key": api_key}
    )
    if response.status_code != 200:
        raise YouTubeAPIError(path, response.status_code, response.text)
    return response.json()


# ---------------------------------------------------------------------------
# Channel resolution + enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelInfo:
    channel_id: str
    title: str
    uploads_playlist_id: str
    video_count: int


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    comment_count: int


@dataclass(frozen=True)
class QuotaEstimate:
    """Pre-flight result: how much this analysis would cost, computed
    without pulling a single comment (SPEC §4.4: refuse *before* spending).
    """
    channel: ChannelInfo
    videos: list[VideoMeta]
    total_comment_count: int
    units_already_spent_on_estimate: int
    units_required_for_comment_pull: int

    @property
    def total_units_required(self) -> int:
        return self.units_already_spent_on_estimate + self.units_required_for_comment_pull


async def _resolve_channel(
    parsed: ParsedInput, *, client: httpx.AsyncClient, api_key: str, ledger: QuotaLedger,
) -> ChannelInfo:
    if parsed.kind == "video":
        ledger.require(COST_VIDEOS_LIST)
        data = await _get(
            client, "videos", {"part": "snippet", "id": parsed.video_id}, api_key
        )
        ledger.charge(COST_VIDEOS_LIST)
        items = data.get("items") or []
        if not items:
            raise ChannelNotFoundError(f"No video found for id={parsed.video_id!r}")
        channel_id = items[0]["snippet"]["channelId"]
        return await _fetch_channel_by(client, api_key, ledger, id=channel_id)

    if parsed.kind == "channel_id":
        return await _fetch_channel_by(client, api_key, ledger, id=parsed.channel_id)
    if parsed.kind == "handle":
        return await _fetch_channel_by(client, api_key, ledger, forHandle=parsed.handle)
    if parsed.kind == "legacy_username":
        return await _fetch_channel_by(client, api_key, ledger, forUsername=parsed.legacy_username)
    raise AssertionError(f"unhandled ParsedInput.kind: {parsed.kind!r}")  # exhaustive


async def _fetch_channel_by(
    client: httpx.AsyncClient, api_key: str, ledger: QuotaLedger, **id_param: str
) -> ChannelInfo:
    ledger.require(COST_CHANNELS_LIST)
    data = await _get(
        client, "channels", {"part": "contentDetails,statistics,snippet", **id_param}, api_key
    )
    ledger.charge(COST_CHANNELS_LIST)
    items = data.get("items") or []
    if not items:
        raise ChannelNotFoundError(f"No channel found for {id_param!r}")
    item = items[0]
    return ChannelInfo(
        channel_id=item["id"],
        title=item["snippet"]["title"],
        uploads_playlist_id=item["contentDetails"]["relatedPlaylists"]["uploads"],
        video_count=int(item.get("statistics", {}).get("videoCount", 0)),
    )


async def _enumerate_video_ids(
    client: httpx.AsyncClient,
    api_key: str,
    ledger: QuotaLedger,
    uploads_playlist_id: str,
    max_videos: int,
) -> list[str]:
    """Most-recent-first video IDs from a channel's uploads playlist."""
    video_ids: list[str] = []
    page_token: str | None = None
    while len(video_ids) < max_videos:
        ledger.require(COST_PLAYLIST_ITEMS_LIST)
        params: dict = {
            "part": "contentDetails",
            "playlistId": uploads_playlist_id,
            "maxResults": min(PLAYLIST_ITEMS_PAGE_SIZE, max_videos - len(video_ids)),
        }
        if page_token:
            params["pageToken"] = page_token
        data = await _get(client, "playlistItems", params, api_key)
        ledger.charge(COST_PLAYLIST_ITEMS_LIST)
        video_ids.extend(item["contentDetails"]["videoId"] for item in data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:max_videos]


async def _fetch_video_metas(
    client: httpx.AsyncClient, api_key: str, ledger: QuotaLedger, video_ids: list[str],
) -> list[VideoMeta]:
    """Batched videos.list lookup (up to 50 ids/call) for title + commentCount."""
    metas: list[VideoMeta] = []
    for start in range(0, len(video_ids), VIDEOS_BATCH_SIZE):
        batch = video_ids[start : start + VIDEOS_BATCH_SIZE]
        ledger.require(COST_VIDEOS_LIST)
        data = await _get(
            client, "videos", {"part": "snippet,statistics", "id": ",".join(batch)}, api_key
        )
        ledger.charge(COST_VIDEOS_LIST)
        by_id = {item["id"]: item for item in data.get("items", [])}
        for video_id in batch:
            item = by_id.get(video_id)
            if item is None:
                continue  # deleted/private since enumeration — skip rather than half-fail
            metas.append(
                VideoMeta(
                    video_id=video_id,
                    title=item["snippet"]["title"],
                    comment_count=int(item.get("statistics", {}).get("commentCount", 0)),
                )
            )
    return metas


async def estimate_channel_analysis(
    url_or_id: str,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    ledger: QuotaLedger,
    max_videos: int = DEFAULT_MAX_VIDEOS,
) -> QuotaEstimate:
    """SPEC §4.4 pre-flight: resolve the channel, enumerate up to
    *max_videos* most-recent uploads, and compute the exact quota cost of
    pulling every comment on all of them — without pulling a single one.

    Raises QuotaExceededError immediately if even this cheap estimation
    pass can't be afforded; the caller should check
    `estimate.units_required_for_comment_pull` against
    `ledger.remaining` before calling `fetch_channel_comments`.
    """
    parsed = parse_youtube_url(url_or_id)
    spent_before = ledger.spent
    channel = await _resolve_channel(parsed, client=client, api_key=api_key, ledger=ledger)
    video_ids = await _enumerate_video_ids(
        client, api_key, ledger, channel.uploads_playlist_id, max_videos
    )
    videos = await _fetch_video_metas(client, api_key, ledger, video_ids)
    total_comments = sum(v.comment_count for v in videos)
    pull_cost = sum(_comment_pages(v.comment_count) for v in videos)
    return QuotaEstimate(
        channel=channel,
        videos=videos,
        total_comment_count=total_comments,
        units_already_spent_on_estimate=ledger.spent - spent_before,
        units_required_for_comment_pull=pull_cost,
    )


def _comment_pages(comment_count: int) -> int:
    """commentThreads.list pages (1 unit each) needed for *comment_count*
    top-level threads, at COMMENT_THREADS_PAGE_SIZE per page."""
    if comment_count <= 0:
        return 0
    return -(-comment_count // COMMENT_THREADS_PAGE_SIZE)  # ceil division


# ---------------------------------------------------------------------------
# The actual comment pull
# ---------------------------------------------------------------------------

def _comment_from_snippet(
    comment_id: str, video_id: str, snippet: dict, *, is_reply: bool
) -> RawComment | None:
    text = snippet.get("textOriginal") or snippet.get("textDisplay")
    if not text:
        return None
    published_at = snippet.get("publishedAt")
    timestamp = (
        datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if published_at
        else datetime.now(timezone.utc)
    )
    return RawComment(
        id=comment_id,
        platform="youtube",
        text=sanitize_comment_text(text),
        author_handle=snippet.get("authorDisplayName"),
        author_id=(snippet.get("authorChannelId") or {}).get("value"),
        timestamp=timestamp,
        video_id=video_id,
        like_count=int(snippet.get("likeCount", 0)),
        is_reply=is_reply,
    )


async def fetch_video_comments(
    video_id: str,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    ledger: QuotaLedger,
    dedup: ContentDeduplicator | None = None,
) -> AsyncIterator[RawComment]:
    """Paginate commentThreads.list for one video, yielding normalized,
    deduplicated RawComment records (top-level comments and their inline
    replies — both come back in the same page, so replies cost nothing
    extra). Charges COST_COMMENT_THREADS_LIST per page *before* the request
    and raises QuotaExceededError rather than making a call it can't afford
    — never a partial pull that then dies mid-video.
    """
    page_token: str | None = None
    while True:
        ledger.require(COST_COMMENT_THREADS_LIST)
        params: dict = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": COMMENT_THREADS_PAGE_SIZE,
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            data = await _get(client, "commentThreads", params, api_key)
        except YouTubeAPIError as exc:
            if exc.status_code == 403 and page_token is None:
                # Comments disabled on this video. One video failing this
                # way shouldn't kill an entire channel analysis (SPEC §4.4:
                # "never half-fail") — skip it and keep going. The quota
                # unit was never charged for this call since it never
                # succeeded.
                logger.warning("Comments disabled or forbidden on video %s; skipping", video_id)
                return
            raise
        ledger.charge(COST_COMMENT_THREADS_LIST)

        for thread in data.get("items", []):
            top = thread["snippet"]["topLevelComment"]
            comment = _comment_from_snippet(top["id"], video_id, top["snippet"], is_reply=False)
            comment = await _dedup_pass(comment, dedup)
            if comment is not None:
                yield comment
            for reply in (thread.get("replies") or {}).get("comments", []):
                reply_comment = _comment_from_snippet(
                    reply["id"], video_id, reply["snippet"], is_reply=True
                )
                reply_comment = await _dedup_pass(reply_comment, dedup)
                if reply_comment is not None:
                    yield reply_comment

        page_token = data.get("nextPageToken")
        if not page_token:
            break


async def _dedup_pass(
    comment: RawComment | None, dedup: ContentDeduplicator | None
) -> RawComment | None:
    """Return *comment* unless it's already a known duplicate (or None)."""
    if comment is None or dedup is None:
        return comment
    fp = fingerprint_text(comment.text)
    if await dedup.is_duplicate(fp):
        return None
    return comment


async def fetch_channel_comments(
    estimate: QuotaEstimate,
    *,
    client: httpx.AsyncClient,
    api_key: str,
    ledger: QuotaLedger,
    dedup: ContentDeduplicator | None = None,
) -> AsyncIterator[RawComment]:
    """Pull every comment for every video in *estimate*, most-recent video
    first. Call `estimate_channel_analysis` first and check
    `estimate.units_required_for_comment_pull <= ledger.remaining` (or just
    let this raise QuotaExceededError up front on the first page) — SPEC
    §4.4: refuse before spending, never half-fail partway through a video.
    """
    ledger.require(estimate.units_required_for_comment_pull)
    for video in estimate.videos:
        if video.comment_count <= 0:
            # The pre-flight estimate charged zero units for this video
            # (SPEC §4.4: quota must be computed, not guessed) — honour
            # that by not spending a call on it either.
            continue
        async for comment in fetch_video_comments(
            video.video_id, client=client, api_key=api_key, ledger=ledger, dedup=dedup
        ):
            yield comment
