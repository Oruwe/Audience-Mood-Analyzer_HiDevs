"""L1 unit tests — schemas.py's Stage C / SPEC §3 insight-contract models
and the shared _validate_quotes_verbatim helper, in isolation from any LLM
call. The "hard test" SPEC §10 invariant 3 asks for (a fabricated quote
against a real synthesis call) lives in
tests/test_l2_invariant3_verbatim_quotes.py — this file is the schema-level
plumbing underneath it.
"""

import pytest
from pydantic import ValidationError

from schemas import (
    ChannelInsights,
    ConfusionInsight,
    ConfusionInsightDraft,
    RequestInsight,
    RequestInsightDraft,
    VideoDriverDraft,
    VideoMoodInsight,
)

CORPUS = {
    "Please make a Docker follow-up video!",
    "I second the Docker request, would love that",
    "Lost me at the env var setup, what does PORT do?",
}


def test_valid_verbatim_quotes_pass():
    draft = RequestInsightDraft.model_validate(
        {
            "theme": "Wants a Docker follow-up",
            "quotes": ["Please make a Docker follow-up video!", "I second the Docker request, would love that"],
            "suggested_title": "Docker for Beginners",
        },
        context={"corpus": CORPUS},
    )
    assert len(draft.quotes) == 2


def test_fabricated_quote_is_rejected_with_context():
    with pytest.raises(ValidationError, match="verbatim"):
        RequestInsightDraft.model_validate(
            {
                "theme": "Wants a Docker follow-up",
                "quotes": ["Please make a Docker follow-up video!", "This quote was never said by anyone"],
                "suggested_title": "Docker for Beginners",
            },
            context={"corpus": CORPUS},
        )


def test_partial_verbatim_substring_is_accepted():
    """"Verbatim" means an exact excerpt, not necessarily the whole
    comment -- a quote can be a real substring of a longer comment."""
    draft = ConfusionInsightDraft.model_validate(
        {
            "sticking_point": "Confused about the PORT env var",
            "quotes": ["what does PORT do?"],
            "timestamp_hint": None,
        },
        context={"corpus": CORPUS},
    )
    assert draft.quotes == ["what does PORT do?"]


def test_missing_context_skips_validation():
    """No context -- e.g. reconstructing an already-checkpointed result
    from Postgres -- must not fail just because the corpus wasn't handed
    back in. This is a documented escape hatch, not a loophole in the
    actual synthesis path (which always supplies a corpus)."""
    draft = RequestInsightDraft.model_validate({
        "theme": "x",
        "quotes": ["anything at all", "even fabricated text"],
        "suggested_title": "y",
    })
    assert draft.quotes == ["anything at all", "even fabricated text"]


@pytest.mark.parametrize("quotes", [[], ["only one"], ["a", "b", "c", "d"]])
def test_request_insight_requires_two_or_three_quotes(quotes):
    with pytest.raises(ValidationError):
        RequestInsightDraft.model_validate(
            {"theme": "x", "quotes": quotes, "suggested_title": "y"},
        )


def test_confusion_insight_allows_one_to_five_quotes():
    draft = ConfusionInsightDraft.model_validate({
        "sticking_point": "x", "quotes": ["one real quote"], "timestamp_hint": None,
    })
    assert len(draft.quotes) == 1


def test_video_driver_draft_requires_one_to_three_quotes():
    with pytest.raises(ValidationError):
        VideoDriverDraft.model_validate({"top_negative_driver": "x", "quotes": []})


def test_mention_count_is_a_real_field_on_the_final_models_not_the_draft():
    assert "mention_count" not in RequestInsightDraft.model_fields
    assert "mention_count" in RequestInsight.model_fields
    assert "mention_count" not in ConfusionInsightDraft.model_fields
    assert "mention_count" in ConfusionInsight.model_fields


def test_channel_insights_composes_all_three_blocks():
    insights = ChannelInsights(
        requests=[RequestInsight(
            theme="t", quotes=["a", "b"], suggested_title="s", mention_count=2,
        )],
        confusion_points=[ConfusionInsight(
            sticking_point="p", quotes=["c"], timestamp_hint=None, mention_count=1,
        )],
        video_moods=[VideoMoodInsight(
            video_title="v", sentiment_score=-0.3, delta_vs_channel_avg=-0.5,
            top_negative_driver="d", quotes=["e"],
        )],
    )
    assert len(insights.requests) == 1
    assert len(insights.confusion_points) == 1
    assert len(insights.video_moods) == 1
