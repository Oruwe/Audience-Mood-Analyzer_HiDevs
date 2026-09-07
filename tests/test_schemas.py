"""Tests for the shared Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from schemas import DeepMoodAnalysis, PrimaryIntent, RawComment, RecommendedAction, Sentiment


def _base_analysis(**overrides) -> dict:
    payload = {
        "sentiment": Sentiment.POSITIVE,
        "confidence": 0.5,
        "primary_intent": PrimaryIntent.GENERAL_INQUIRY,
        "urgency_score": 0.5,
        "summary": "ok",
        "recommended_action": RecommendedAction.IGNORE,
    }
    payload.update(overrides)
    return payload


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DeepMoodAnalysis(**_base_analysis(confidence=1.5))


def test_urgency_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DeepMoodAnalysis(**_base_analysis(urgency_score=-0.1))


def test_emotional_drivers_defaults_to_empty_list():
    analysis = DeepMoodAnalysis(**_base_analysis())
    assert analysis.emotional_drivers == []


def test_brand_safety_flag_defaults_to_false():
    analysis = DeepMoodAnalysis(**_base_analysis())
    assert analysis.brand_safety_flag is False


def test_raw_comment_requires_core_fields():
    with pytest.raises(ValidationError):
        RawComment(id="x", platform="twitter")  # missing text/timestamp


def test_raw_comment_author_fields_are_optional():
    comment = RawComment(
        id="x", platform="twitter", text="hi", timestamp=datetime.now(UTC)
    )
    assert comment.author_handle is None
    assert comment.author_id is None
