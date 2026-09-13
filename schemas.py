"""Pydantic schemas shared across ingestion, analysis, and storage layers."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ValidationInfo, field_validator


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


# ---------------------------------------------------------------------------
# Stage C / SPEC §3 — the insight contract. "Every single one must cite
# verbatim comments." Enforced here, in the schema (SPEC §10 invariant 3),
# not by asking the model nicely in the prompt: every `quotes` field runs
# through _validate_quotes_verbatim, which checks each quote is an exact
# substring of some comment's text — the pool of allowed text is passed in
# explicitly as `context={"corpus": {...}}` to model_validate/
# model_validate_json. Omitting context skips the check entirely (used
# only for reconstructing an already-validated object, e.g. loading a
# checkpointed result back out of Postgres — never for a fresh LLM
# response, which must always be validated with a real corpus).
# ---------------------------------------------------------------------------

def _validate_quotes_verbatim(quotes: list[str], info: ValidationInfo) -> list[str]:
    context = info.context or {}
    corpus: set[str] | None = context.get("corpus")
    if corpus is None:
        return quotes
    fabricated = [q for q in quotes if not any(q in text for text in corpus)]
    if fabricated:
        raise ValueError(
            "Quote(s) are not verbatim excerpts of any comment in the "
            f"corpus (SPEC §10 invariant 3): {fabricated!r}"
        )
    return quotes


class RequestInsightDraft(BaseModel):
    """LLM-facing contract for one Stage C 'requests' cluster call."""
    theme: str
    quotes: list[str] = Field(min_length=2, max_length=3)  # SPEC §3: "2-3 verbatim, unedited"
    suggested_title: str

    @field_validator("quotes")
    @classmethod
    def _quotes_verbatim(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _validate_quotes_verbatim(v, info)


class RequestInsight(RequestInsightDraft):
    """SPEC §3 Block 1. mention_count is computed from the source cluster
    by engine/insights.py, never trusted from the model — it's a fact
    already known exactly, not something to ask an LLM to count."""
    mention_count: int = Field(ge=1)


class ConfusionInsightDraft(BaseModel):
    """LLM-facing contract for one Stage C 'confusion' cluster call."""
    sticking_point: str
    quotes: list[str] = Field(min_length=1, max_length=5)
    timestamp_hint: str | None = None

    @field_validator("quotes")
    @classmethod
    def _quotes_verbatim(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _validate_quotes_verbatim(v, info)


class ConfusionInsight(ConfusionInsightDraft):
    """SPEC §3 Block 2. mention_count computed, as RequestInsight."""
    mention_count: int = Field(ge=1)


class VideoDriverDraft(BaseModel):
    """LLM-facing contract for one Stage C 'underperforming video' call —
    only the part that genuinely needs language understanding."""
    top_negative_driver: str
    quotes: list[str] = Field(min_length=1, max_length=3)

    @field_validator("quotes")
    @classmethod
    def _quotes_verbatim(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _validate_quotes_verbatim(v, info)


class VideoMoodInsight(BaseModel):
    """SPEC §3 Block 3. sentiment_score/delta_vs_channel_avg are pure Stage
    A aggregation (engine/insights.py) — no LLM call, no synthesis task.
    Only top_negative_driver + quotes come from the model, and only for
    videos that actually underperformed (SPEC: "which video underperformed
    emotionally, and why" — not a row for every video)."""
    video_title: str
    sentiment_score: float
    delta_vs_channel_avg: float
    top_negative_driver: str
    quotes: list[str]

    @field_validator("quotes")
    @classmethod
    def _quotes_verbatim(cls, v: list[str], info: ValidationInfo) -> list[str]:
        return _validate_quotes_verbatim(v, info)


class ChannelInsights(BaseModel):
    """The full SPEC §3 report for one channel analysis."""
    requests: list[RequestInsight]
    confusion_points: list[ConfusionInsight]
    video_moods: list[VideoMoodInsight]
