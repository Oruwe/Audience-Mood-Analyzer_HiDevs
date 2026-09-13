"""Pydantic schemas shared across ingestion, analysis, and storage layers."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Sentiment(str, Enum):
    STRONGLY_POSITIVE = "strongly_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    CRITICAL_ESCALATION = "critical_escalation"


class PrimaryIntent(str, Enum):
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    PRICING_COMPLAINT = "pricing_complaint"
    PRAISE_ENDORSEMENT = "praise_endorsement"
    CHURN_RISK = "churn_risk"
    GENERAL_INQUIRY = "general_inquiry"
    SARCASTIC_TROLL = "sarcastic_troll"


class RecommendedAction(str, Enum):
    IGNORE = "ignore"
    COMMUNITY_REPLY = "community_reply"
    ESCALATE_TO_SUPPORT = "escalate_to_support"
    ESCALATE_TO_PR = "escalate_to_pr"
    AMPLIFY_MARKETING = "amplify_marketing"


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


class DeepMoodAnalysis(BaseModel):
    """LLM-facing contract: only the fields the model must generate."""
    sentiment: Sentiment
    confidence: float = Field(ge=0.0, le=1.0)
    primary_intent: PrimaryIntent
    urgency_score: float = Field(ge=0.0, le=1.0)
    emotional_drivers: list[str] = Field(default_factory=list)
    summary: str
    recommended_action: RecommendedAction
    suggested_reply_draft: str | None = None
    brand_safety_flag: bool = False


class EnrichedCommentRecord(DeepMoodAnalysis):
    """Full analysis record. (Persistence target is being rebuilt per
    SPEC §4.3 — Postgres/Neon, not the DuckDB this record's shape predates.)
    """
    comment_id: str
    platform: str | None = None
    author_handle: str | None = None
    raw_text: str | None = None
    embedding: list[float] | None = None
    cluster_id: int | None = None
    latency_ms: float | None = None
    model_used: str | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
