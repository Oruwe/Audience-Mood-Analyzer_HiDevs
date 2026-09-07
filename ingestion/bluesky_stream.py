"""Live ingestion: Bluesky public firehose via the Jetstream WebSocket."""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlencode

import websockets
from pydantic import BaseModel, ConfigDict

from schemas import RawComment

logger = logging.getLogger(__name__)

JETSTREAM_URL = "wss://jetstream1.us-east.bsky.network/subscribe"
DEFAULT_KEYWORDS = ["ai", "tech", "marketing", "launch"]


class JetstreamRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str = ""
    createdAt: str | None = None
    langs: list[str] = []


class JetstreamCommit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    operation: str
    collection: str
    rkey: str
    record: JetstreamRecord


class JetstreamMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    did: str
    time_us: int
    kind: str
    commit: JetstreamCommit | None = None


def _build_url(langs: list[str] | None) -> str:
    params = {"wantedCollections": "app.bsky.feed.post"}
    if langs:
        params["wantedLangs"] = ",".join(langs)  # server-side filter
    return f"{JETSTREAM_URL}?{urlencode(params)}"


def _matches(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _to_raw_comment(msg: JetstreamMessage) -> RawComment:
    commit = msg.commit
    assert commit is not None
    if commit.record.createdAt:
        ts = datetime.fromisoformat(commit.record.createdAt.replace("Z", "+00:00"))
    else:
        ts = datetime.fromtimestamp(msg.time_us / 1_000_000, tz=UTC)
    return RawComment(
        id=f"bsky:{msg.did}:{commit.rkey}",
        platform="bluesky",
        text=commit.record.text,
        author_id=msg.did,  # Jetstream emits DIDs; handle resolution out of scope
        timestamp=ts,
    )


async def generate_bluesky_stream(
    keywords: list[str] | None = None,
    langs: list[str] | None = None,
) -> AsyncIterator[RawComment]:
    """Consume the public Jetstream firehose forever, auto-reconnecting."""
    keywords = keywords or DEFAULT_KEYWORDS
    url = _build_url(langs)
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, max_size=2**22) as ws:
                logger.info("Jetstream connected (keywords=%s, langs=%s)", keywords, langs)
                backoff = 1.0
                async for payload in ws:
                    try:
                        msg = JetstreamMessage.model_validate_json(payload)
                    except ValueError:
                        continue  # malformed event — skip, never kill the stream
                    commit = msg.commit
                    if msg.kind != "commit" or commit is None or commit.operation != "create":
                        continue
                    if not commit.record.text or not _matches(commit.record.text, keywords):
                        continue
                    yield _to_raw_comment(msg)
        except Exception as exc:  # noqa: BLE001 — firehose must survive anything
            logger.warning("Jetstream error (%s); reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
