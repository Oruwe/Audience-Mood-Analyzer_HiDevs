"""L2 contract tests — engine.insights's build_requests / build_confusion_points
/ build_video_moods, against a stubbed litellm.acompletion (no network, no
OpenRouter key). KMeans is real here (small, well-separated toy embeddings)
since clustering-mechanics coverage already lives in
tests/test_l1_insights_clustering.py; these tests are about candidate
filtering, mention_count computation, and per-video stats.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import engine.insights as insights
from schemas import CommentIntent, RawComment, Sentiment, StageASentimentItem, StageBClassificationItem

API_KEY = "fake-key"


def _comment(cid: str, text: str, video_id: str = "v1") -> RawComment:
    return RawComment(
        id=cid, platform="youtube", text=text,
        timestamp=datetime.now(timezone.utc), video_id=video_id,
    )


def _stage_b(comment_id: str, *, is_request=False, is_confusion=False) -> StageBClassificationItem:
    intent = CommentIntent.REQUEST if is_request else (
        CommentIntent.CONFUSION if is_confusion else CommentIntent.OTHER
    )
    return StageBClassificationItem(
        comment_id=comment_id, intent=intent, is_request=is_request, is_confusion=is_confusion,
    )


def _fake_response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


def _run(coro):
    return asyncio.run(coro)


def test_build_requests_only_considers_is_request_flagged_comments(monkeypatch):
    comments = [
        _comment("c0", "Please do a Docker follow-up video"),
        _comment("c1", "I agree, Docker would be great"),
        _comment("c2", "Great video, thanks!"),  # not a request -- must be excluded
    ]
    stage_b = {
        "c0": _stage_b("c0", is_request=True),
        "c1": _stage_b("c1", is_request=True),
        "c2": _stage_b("c2", is_request=False),
    }
    embeddings = {"c0": [0.0, 0.0], "c1": [0.01, 0.0], "c2": [9.0, 9.0]}

    seen_ids = []

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        content = messages[1]["content"]
        seen_ids.extend(line.split(":")[0] for line in content.splitlines())
        return _fake_response({
            "theme": "Wants a Docker follow-up",
            "quotes": ["Please do a Docker follow-up video", "I agree, Docker would be great"],
            "suggested_title": "Docker 101",
        })

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)

    result = _run(insights.build_requests(comments, stage_b, embeddings, api_key=API_KEY))

    assert "c2" not in seen_ids
    assert len(result) == 1
    assert result[0].mention_count == 2  # computed from cluster size, not the model
    assert result[0].theme == "Wants a Docker follow-up"


def test_build_confusion_points_only_considers_is_confusion_flagged_comments(monkeypatch):
    comments = [
        _comment("c0", "Lost me at the env var step"),
        _comment("c1", "Yeah what does PORT do"),
    ]
    stage_b = {
        "c0": _stage_b("c0", is_confusion=True),
        "c1": _stage_b("c1", is_confusion=True),
    }
    embeddings = {"c0": [0.0, 0.0], "c1": [0.01, 0.0]}

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        return _fake_response({
            "sticking_point": "Confused about env vars",
            "quotes": ["Lost me at the env var step"],
            "timestamp_hint": None,
        })

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)

    result = _run(insights.build_confusion_points(comments, stage_b, embeddings, api_key=API_KEY))

    assert len(result) == 1
    assert result[0].mention_count == 2
    assert result[0].sticking_point == "Confused about env vars"


def test_a_failed_cluster_is_skipped_not_fatal(monkeypatch):
    comments = [
        _comment("c0", "Please make a follow-up"),
        _comment("c1", "Same, would love a follow-up"),
    ]
    stage_b = {"c0": _stage_b("c0", is_request=True), "c1": _stage_b("c1", is_request=True)}
    embeddings = {"c0": [0.0, 0.0], "c1": [0.01, 0.0]}

    async def broken_acompletion(*, model, api_key, messages, response_format, timeout):
        return _fake_response({"not": "the right shape at all"})

    monkeypatch.setattr(insights, "acompletion", broken_acompletion)

    result = _run(insights.build_requests(comments, stage_b, embeddings, api_key=API_KEY))

    assert result == []  # skipped, no exception propagated


def test_build_video_moods_computes_sentiment_stats_without_any_llm_call_for_non_underperformers(monkeypatch):
    comments = [
        _comment("c0", "loved it", video_id="good-video"),
        _comment("c1", "amazing", video_id="good-video"),
        _comment("c2", "terrible", video_id="bad-video"),
        _comment("c3", "awful, so confusing", video_id="bad-video"),
    ]
    sentiments = {
        "c0": StageASentimentItem(comment_id="c0", sentiment=Sentiment.STRONGLY_POSITIVE, confidence=0.9),
        "c1": StageASentimentItem(comment_id="c1", sentiment=Sentiment.STRONGLY_POSITIVE, confidence=0.9),
        "c2": StageASentimentItem(comment_id="c2", sentiment=Sentiment.NEGATIVE, confidence=0.9),
        "c3": StageASentimentItem(comment_id="c3", sentiment=Sentiment.CRITICAL_ESCALATION, confidence=0.9),
    }
    video_titles = {"good-video": "The Good One", "bad-video": "The Bad One"}
    calls = {"count": 0}

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        calls["count"] += 1
        return _fake_response({
            "top_negative_driver": "Confusing setup steps",
            "quotes": ["awful, so confusing"],
        })

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)

    result = _run(insights.build_video_moods(comments, sentiments, video_titles, api_key=API_KEY))

    # Only "bad-video" underperforms (negative delta) -- exactly one call made.
    assert calls["count"] == 1
    assert len(result) == 1
    mood = result[0]
    assert mood.video_title == "The Bad One"
    assert mood.delta_vs_channel_avg < 0
    assert mood.top_negative_driver == "Confusing setup steps"


def test_build_video_moods_never_reports_a_non_underperforming_video(monkeypatch):
    comments = [
        _comment("c0", "loved it", video_id="only-video"),
        _comment("c1", "amazing", video_id="only-video"),
    ]
    sentiments = {
        "c0": StageASentimentItem(comment_id="c0", sentiment=Sentiment.STRONGLY_POSITIVE, confidence=0.9),
        "c1": StageASentimentItem(comment_id="c1", sentiment=Sentiment.STRONGLY_POSITIVE, confidence=0.9),
    }

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("no video underperforms here -- must not call the model")

    monkeypatch.setattr(insights, "acompletion", fail_if_called)

    # A single video is always exactly at the channel average (delta 0) --
    # never negative, so it must never appear in the output.
    result = _run(insights.build_video_moods(
        comments, sentiments, {"only-video": "Only Video"}, api_key=API_KEY
    ))
    assert result == []
