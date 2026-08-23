"""Pydantic schemas shared across ingestion, analysis, and storage layers."""

from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

class Mood(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class MarketingAction(str, Enum):
    NONE = "none"
    REPLY = "reply"
    AMPLIFY = "amplify"
    ESCALATE = "escalate"

class RawComment(BaseModel):
    """A single inbound social-media comment, pre-analysis."""
    id: str
    platform: str
    text: str
    author: str
    timestamp: datetime

class MoodAnalysis(BaseModel):
    """LLM-facing contract: only the fields the model must generate."""
    mood: Mood
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    urgency_score: float = Field(ge=0.0, le=1.0)
    marketing_action: MarketingAction

class AnalyzedMood(MoodAnalysis):
    """Full analysis record persisted to DuckDB."""
    comment_id: str
    platform: str | None = None
    author: str | None = None
    comment_text: str | None = None
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
