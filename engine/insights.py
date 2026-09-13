"""SPEC §3 + §4.1 Stage C: synthesis — the three insight blocks, each
citing verbatim comments. Evidence citation is enforced in the schema
itself (schemas.py's `_validate_quotes_verbatim`, SPEC §10 invariant 3),
not by asking the model nicely in the prompt: every draft response is
parsed via `model_validate_json(..., context={"corpus": {...}})`, and a
quote that isn't an exact substring of some comment in that corpus fails
validation before it can reach anything downstream.

Stage C's "~8 calls" budget (SPEC §4.1) is spent only where genuine
language understanding earns its place: naming a theme, drafting a video
title, judging whether commenters cited a timestamp, and explaining *why*
a video underperformed. Block 3's sentiment_score/delta_vs_channel_avg are
pure arithmetic over Stage A's already-computed sentiment — there is no
synthesis task there an LLM would do better than a mean, so no call is
spent on it.

Budget split (a documented starting point, not a measured optimum — same
caveat as engine/stage_filter.py's thresholds; there's no channel data yet
to tune against):
  up to 3 calls — Requests clusters (comments flagged is_request)
  up to 3 calls — Confusion clusters (comments flagged is_confusion)
  up to 2 calls — the worst-performing videos' top_negative_driver
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import openai
from litellm import acompletion
from pydantic import BaseModel, ValidationError
from sklearn.cluster import KMeans

from config.models import STAGE_C_SYNTHESIS, STAGE_C_SYNTHESIS_FALLBACK
from resilience import is_fallback_worthy_api_error, retry_transient_api_error
from schemas import (
    ChannelInsights,
    ConfusionInsight,
    ConfusionInsightDraft,
    RawComment,
    RequestInsight,
    RequestInsightDraft,
    Sentiment,
    StageASentimentItem,
    StageBClassificationItem,
    VideoDriverDraft,
    VideoMoodInsight,
)

logger = logging.getLogger(__name__)

MAX_REQUEST_CLUSTERS = 3
MAX_CONFUSION_CLUSTERS = 3
MAX_UNDERPERFORMING_VIDEOS = 2
MIN_CLUSTER_SIZE = 2  # a "theme" one person mentioned once isn't a theme
DRIVER_SAMPLE_SIZE = 20  # cap the LLM's input to a video's worst N comments

_SENTIMENT_SCORE: dict[Sentiment, float] = {
    Sentiment.STRONGLY_POSITIVE: 1.0,
    Sentiment.POSITIVE: 0.5,
    Sentiment.NEUTRAL: 0.0,
    Sentiment.NEGATIVE: -0.5,
    Sentiment.CRITICAL_ESCALATION: -1.0,
}

REQUEST_SYSTEM_PROMPT = (
    "You are reading a cluster of YouTube comments that all ask the "
    "creator for future content. Respond ONLY with JSON matching exactly "
    'this schema: {"theme": "<short phrase, e.g. \'Wants a Docker '
    'follow-up\'>", "quotes": ["<verbatim comment text>", "..."], '
    '"suggested_title": "<a video title that answers the request>"}. '
    "quotes must be 2 or 3 items, each copied EXACTLY as written in the "
    "comments given — do not paraphrase, correct, or shorten them."
)

CONFUSION_SYSTEM_PROMPT = (
    "You are reading a cluster of YouTube comments that all express "
    "confusion about something in the creator's video. Respond ONLY with "
    'JSON matching exactly this schema: {"sticking_point": "<short phrase, '
    "e.g. 'Lost people at the env var setup'>\", "
    '"quotes": ["<verbatim comment text>", "..."], '
    '"timestamp_hint": "<a timestamp commenters mentioned, or null>"}. '
    "quotes must be 1 to 5 items, each copied EXACTLY as written in the "
    "comments given — do not paraphrase, correct, or shorten them."
)

DRIVER_SYSTEM_PROMPT = (
    "You are reading a video's most negative YouTube comments. Respond "
    'ONLY with JSON matching exactly this schema: {"top_negative_driver": '
    '"<short phrase describing what upset viewers>", '
    '"quotes": ["<verbatim comment text>", "..."]}. '
    "quotes must be 1 to 3 items, each copied EXACTLY as written in the "
    "comments given — do not paraphrase, correct, or shorten them."
)


@retry_transient_api_error()
async def _call_stage_c(model: str, api_key: str, messages: list[dict], response_schema: type[BaseModel]):
    return await acompletion(
        model=model, api_key=api_key, messages=messages,
        response_format=response_schema, timeout=30,
    )


async def _call_stage_c_with_fallback(
    models: tuple[str, ...], api_key: str, messages: list[dict], response_schema: type[BaseModel],
):
    """Same reasoning as engine.batching._call_model_with_fallback: a free
    OpenRouter model's transient failure -- a 429, an "Nvidia: Service
    temporarily overloaded" surfaced as a bare litellm.APIError, or a 404
    because OpenRouter withdrew the `:free` slug entirely -- can mean this
    model won't work at all; try the next one, from a different vendor,
    before giving up on this cluster.
    """
    last_exc: Exception | None = None
    for i, model in enumerate(models):
        try:
            return await _call_stage_c(model, api_key, messages, response_schema)
        except openai.APIError as exc:
            if not is_fallback_worthy_api_error(exc):
                raise
            last_exc = exc
            if i + 1 < len(models):
                logger.warning(
                    "Stage C: %s unavailable after retries (%s); falling back to %s",
                    model, exc, models[i + 1],
                )
    assert last_exc is not None  # unreachable with a non-empty models tuple
    raise last_exc


def _cluster_count(n: int, max_clusters: int) -> int:
    if n < MIN_CLUSTER_SIZE:
        return 1
    return max(1, min(max_clusters, n // MIN_CLUSTER_SIZE))


def _cluster_comments(
    comments: list[RawComment], embeddings: dict[str, list[float]], max_clusters: int,
) -> list[list[RawComment]]:
    """Group *comments* into up to *max_clusters* clusters by embedding,
    biggest (most-mentioned) first, dropping any cluster too small to be
    worth a Stage C call. Comments missing an embedding are skipped rather
    than crashing the whole synthesis pass on one gap."""
    usable = [c for c in comments if c.id in embeddings]
    if len(usable) < MIN_CLUSTER_SIZE:
        return []
    k = _cluster_count(len(usable), max_clusters)
    matrix = np.array([embeddings[c.id] for c in usable], dtype=float)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)
    groups: dict[int, list[RawComment]] = {}
    for comment, label in zip(usable, labels):
        groups.setdefault(int(label), []).append(comment)
    big_enough = [g for g in groups.values() if len(g) >= MIN_CLUSTER_SIZE]
    return sorted(big_enough, key=len, reverse=True)[:max_clusters]


async def _synthesize(
    cluster: list[RawComment], *, api_key: str, model: str, system_prompt: str, draft_schema: type[BaseModel],
    fallback_models: tuple[str, ...] = (),
):
    """One Stage C call for one cluster/sample. Returns None (logged, not
    raised) on any failure — one bad cluster shouldn't sink the whole
    report; SPEC's "never half-fail" spirit applied to synthesis rather
    than quota."""
    lines = "\n".join(f"{c.id}: {c.text}" for c in cluster)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": lines},
    ]
    corpus = {c.text for c in cluster}
    try:
        response = await _call_stage_c_with_fallback(
            (model, *fallback_models), api_key, messages, draft_schema
        )
        return draft_schema.model_validate_json(
            response.choices[0].message.content, context={"corpus": corpus}
        )
    except (ValidationError, ValueError) as exc:
        logger.warning(
            "Stage C synthesis failed for a %d-comment cluster (%s): %s",
            len(cluster), draft_schema.__name__, exc,
        )
        return None


async def build_requests(
    comments: list[RawComment],
    stage_b: dict[str, StageBClassificationItem],
    embeddings: dict[str, list[float]],
    *,
    api_key: str,
    model: str = STAGE_C_SYNTHESIS,
    max_clusters: int = MAX_REQUEST_CLUSTERS,
    fallback_models: tuple[str, ...] = (STAGE_C_SYNTHESIS_FALLBACK,),
) -> list[RequestInsight]:
    """SPEC §3 Block 1 — what the audience is asking the creator to make."""
    candidates = [c for c in comments if stage_b.get(c.id) is not None and stage_b[c.id].is_request]
    clusters = _cluster_comments(candidates, embeddings, max_clusters)
    # Independent clusters, independent calls -- run them concurrently
    # rather than one-at-a-time (at most MAX_REQUEST_CLUSTERS=3, so this
    # can't run away on a pathological input).
    drafts = await asyncio.gather(*(
        _synthesize(
            cluster, api_key=api_key, model=model,
            system_prompt=REQUEST_SYSTEM_PROMPT, draft_schema=RequestInsightDraft,
            fallback_models=fallback_models,
        )
        for cluster in clusters
    ))
    return [
        RequestInsight(**draft.model_dump(), mention_count=len(cluster))
        for cluster, draft in zip(clusters, drafts)
        if draft is not None
    ]


async def build_confusion_points(
    comments: list[RawComment],
    stage_b: dict[str, StageBClassificationItem],
    embeddings: dict[str, list[float]],
    *,
    api_key: str,
    model: str = STAGE_C_SYNTHESIS,
    max_clusters: int = MAX_CONFUSION_CLUSTERS,
    fallback_models: tuple[str, ...] = (STAGE_C_SYNTHESIS_FALLBACK,),
) -> list[ConfusionInsight]:
    """SPEC §3 Block 2 — where the creator's explanation didn't land."""
    candidates = [c for c in comments if stage_b.get(c.id) is not None and stage_b[c.id].is_confusion]
    clusters = _cluster_comments(candidates, embeddings, max_clusters)
    drafts = await asyncio.gather(*(
        _synthesize(
            cluster, api_key=api_key, model=model,
            system_prompt=CONFUSION_SYSTEM_PROMPT, draft_schema=ConfusionInsightDraft,
            fallback_models=fallback_models,
        )
        for cluster in clusters
    ))
    return [
        ConfusionInsight(**draft.model_dump(), mention_count=len(cluster))
        for cluster, draft in zip(clusters, drafts)
        if draft is not None
    ]


def _per_video_sentiment_stats(
    comments: list[RawComment], sentiments: dict[str, StageASentimentItem],
) -> dict[str, dict]:
    scores = {
        c.id: _SENTIMENT_SCORE[sentiments[c.id].sentiment]
        for c in comments if c.id in sentiments
    }
    if not scores:
        return {}
    channel_avg = sum(scores.values()) / len(scores)

    by_video: dict[str, list[RawComment]] = {}
    for c in comments:
        if c.video_id and c.id in scores:
            by_video.setdefault(c.video_id, []).append(c)

    stats: dict[str, dict] = {}
    for video_id, video_comments in by_video.items():
        video_avg = sum(scores[c.id] for c in video_comments) / len(video_comments)
        stats[video_id] = {
            "comments": video_comments,
            "sentiment_score": video_avg,
            "delta_vs_channel_avg": video_avg - channel_avg,
        }
    return stats


async def build_video_moods(
    comments: list[RawComment],
    sentiments: dict[str, StageASentimentItem],
    video_titles: dict[str, str],
    *,
    api_key: str,
    model: str = STAGE_C_SYNTHESIS,
    max_videos: int = MAX_UNDERPERFORMING_VIDEOS,
    fallback_models: tuple[str, ...] = (STAGE_C_SYNTHESIS_FALLBACK,),
) -> list[VideoMoodInsight]:
    """SPEC §3 Block 3 — which video underperformed emotionally, and why.
    Only the worst *max_videos* (by delta vs. the channel average, and only
    if that delta is actually negative) get a Stage C call; every video's
    sentiment_score/delta is computed either way, but SPEC asks "which
    video underperformed", not for a row per video.
    """
    stats = _per_video_sentiment_stats(comments, sentiments)
    worst = sorted(stats.items(), key=lambda kv: kv[1]["delta_vs_channel_avg"])
    underperforming = [(vid, s) for vid, s in worst if s["delta_vs_channel_avg"] < 0][:max_videos]

    async def _one(video_id: str, stat: dict):
        sample = sorted(
            stat["comments"], key=lambda c: _SENTIMENT_SCORE[sentiments[c.id].sentiment]
        )[:DRIVER_SAMPLE_SIZE]
        draft = await _synthesize(
            sample, api_key=api_key, model=model,
            system_prompt=DRIVER_SYSTEM_PROMPT, draft_schema=VideoDriverDraft,
            fallback_models=fallback_models,
        )
        if draft is None:
            return None
        return VideoMoodInsight(
            video_title=video_titles.get(video_id, video_id),
            sentiment_score=stat["sentiment_score"],
            delta_vs_channel_avg=stat["delta_vs_channel_avg"],
            top_negative_driver=draft.top_negative_driver,
            quotes=draft.quotes,
        )

    results = await asyncio.gather(*(_one(vid, s) for vid, s in underperforming))
    return [insight for insight in results if insight is not None]


async def build_channel_insights(
    comments: list[RawComment],
    sentiments: dict[str, StageASentimentItem],
    stage_b: dict[str, StageBClassificationItem],
    embeddings: dict[str, list[float]],
    video_titles: dict[str, str],
    *,
    api_key: str,
    model: str = STAGE_C_SYNTHESIS,
    fallback_models: tuple[str, ...] = (STAGE_C_SYNTHESIS_FALLBACK,),
) -> ChannelInsights:
    """The full SPEC §3 report for one channel analysis."""
    # The three blocks are fully independent of each other -- different
    # candidate comments, different prompts, no shared mutable state --
    # so run them concurrently instead of one after another.
    requests, confusion_points, video_moods = await asyncio.gather(
        build_requests(
            comments, stage_b, embeddings, api_key=api_key, model=model, fallback_models=fallback_models,
        ),
        build_confusion_points(
            comments, stage_b, embeddings, api_key=api_key, model=model, fallback_models=fallback_models,
        ),
        build_video_moods(
            comments, sentiments, video_titles, api_key=api_key, model=model, fallback_models=fallback_models,
        ),
    )
    return ChannelInsights(requests=requests, confusion_points=confusion_points, video_moods=video_moods)
