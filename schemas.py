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
    """Full analysis record persisted to DuckDB."""
    comment_id: str
    platform: str | None = None
    author_handle: str | None = None
    raw_text: str | None = None
    embedding: list[float] | None = None
    cluster_id: int | None = None
    latency_ms: float | None = None
    model_used: str | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
