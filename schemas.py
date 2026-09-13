"""Pydantic schemas shared across ingestion, analysis, and storage layers."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    STRONGLY_POSITIVE = "strongly_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CRITICAL_ESCALATION = "critical_escalation"


class RawComment(BaseModel):
    """A single inbound social-media comment, pre-analysis."""
    id: str
    platform: str
    text: str
    author_handle: str | None = None  # e.g. "@maya_builds" (handle-style platforms)
    author_id: str | None = None      # e.g. DID / stable account identifier
    timestamp: datetime
    # --- SPEC §2 "schemas.py — keep, extend" additions for ingestion/youtube.py ---
    video_id: str | None = None       # which video this comment belongs to —
                                       # required for the §3 Block 3 "mood by video" grouping
    like_count: int = 0               # free in the same API response; a cheap salience signal
    is_reply: bool = False            # top-level comment vs. a reply under it


# ---------------------------------------------------------------------------
# Stage A (SPEC §4.1, amended 2026-09-13 — see config/models.py). LLM-facing
# batch contract for sentiment classification over 100% of comments, with
# the §4.1b array-length + id-set guard engine/stage_a.py enforces.
# ---------------------------------------------------------------------------

class StageASentimentItem(BaseModel):
    comment_id: str
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)


class StageASentimentBatch(BaseModel):
    """One LLM response for one batch of comments."""
    results: list[StageASentimentItem]


# ---------------------------------------------------------------------------
# Stage B (SPEC §4.1) — LLM-facing batch contract for the Stage-A-flagged
# subset (~10-20% of comments). Authored fresh for this product: V2's
# PrimaryIntent (bug_report/pricing_complaint/churn_risk/...) was for
# brand-monitoring support triage, which this product isn't. CommentIntent
# below exists to serve SPEC §3's three insight blocks directly.
# ---------------------------------------------------------------------------

class CommentIntent(str, Enum):
    REQUEST = "request"        # asking for future content, e.g. "make a Docker follow-up"
    CONFUSION = "confusion"    # lost/confused about something in the video
    PRAISE = "praise"          # positive reaction, no ask
    CRITICISM = "criticism"    # negative reaction, no ask
    OTHER = "other"            # spam, off-topic, unrelated to the video


class StageBClassificationItem(BaseModel):
    comment_id: str
    intent: CommentIntent
    is_request: bool
    is_confusion: bool


class StageBClassificationBatch(BaseModel):
    """One LLM response for one batch of comments."""
    results: list[StageBClassificationItem]
