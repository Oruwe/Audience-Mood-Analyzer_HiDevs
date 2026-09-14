"""L2 contract test — SPEC §10 invariant 3.

"Every quoted comment is real." Enforced by construction: every Stage C
draft schema (RequestInsightDraft/ConfusionInsightDraft/VideoDriverDraft)
validates its `quotes` field against the exact corpus of comment text it
was synthesized from (schemas.py's _validate_quotes_verbatim), passed in
as `context={"corpus": {...}}` — never trusted from the prompt alone.

This is the "hard test. Fails the build." SPEC §10 asks for: a synthesis
call that returns a fabricated quote must never make it into
engine.insights's output. Stubs litellm.acompletion (no network, no key).
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import engine.insights as insights
from schemas import CommentIntent, RawComment, RequestInsightDraft, StageBClassificationItem

API_KEY = "fake-key"

REAL_COMMENTS = [
    "Please make a Docker follow-up video!",
    "I second the Docker request, would love that",
]

# A batch of ways a model might return text that ISN'T a real quote --
# paraphrased, embellished, or outright invented.
_FABRICATED_QUOTE_ATTEMPTS = [
    "Please make a Docker tutorial series!",       # paraphrased, not verbatim
    "I SECOND THE DOCKER REQUEST, WOULD LOVE THAT",  # case-altered -- "unedited" means exact
    "Everyone in the comments is begging for Docker content",  # fabricated summary, not a quote
    "Please make a Docker follow-up video! Thanks so much!!",  # real quote with an invented addition
]


def _comments() -> list[RawComment]:
    return [
        RawComment(id="c0", platform="youtube", text=REAL_COMMENTS[0],
                   timestamp=datetime.now(timezone.utc), video_id="v1"),
        RawComment(id="c1", platform="youtube", text=REAL_COMMENTS[1],
                   timestamp=datetime.now(timezone.utc), video_id="v1"),
    ]


def _stage_b() -> dict[str, StageBClassificationItem]:
    return {
        "c0": StageBClassificationItem(comment_id="c0", intent=CommentIntent.REQUEST,
                                        is_request=True, is_confusion=False),
        "c1": StageBClassificationItem(comment_id="c1", intent=CommentIntent.REQUEST,
                                        is_request=True, is_confusion=False),
    }


def _embeddings() -> dict[str, list[float]]:
    return {"c0": [0.0, 0.0], "c1": [0.01, 0.0]}


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("fabricated", _FABRICATED_QUOTE_ATTEMPTS)
def test_schema_rejects_every_fabrication_attempt_directly(fabricated):
    with pytest.raises(ValidationError, match="verbatim"):
        RequestInsightDraft.model_validate(
            {
                "theme": "Wants a Docker follow-up",
                "quotes": [REAL_COMMENTS[0], fabricated],
                "suggested_title": "Docker 101",
            },
            context={"corpus": set(REAL_COMMENTS)},
        )


@pytest.mark.parametrize("fabricated", _FABRICATED_QUOTE_ATTEMPTS)
def test_a_synthesis_call_returning_a_fabricated_quote_never_reaches_the_report(monkeypatch, fabricated):
    """The end-to-end guarantee: even if the model actually returns a
    fabricated quote, build_requests must not surface it -- the whole
    cluster is dropped (logged, not raised) rather than shipping a fake
    receipt to a creator."""

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        payload = {
            "theme": "Wants a Docker follow-up",
            "quotes": [REAL_COMMENTS[0], fabricated],
            "suggested_title": "Docker 101",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)

    result = _run(insights.build_requests(_comments(), _stage_b(), _embeddings(), api_key=API_KEY))

    assert result == []  # the fabricated-quote cluster was dropped entirely
    for insight in result:
        for quote in insight.quotes:
            assert quote in REAL_COMMENTS  # would hold trivially since result is empty,
            # but stated explicitly: nothing in a surfaced insight is ever unreal.


def test_a_genuinely_real_quote_still_gets_through(monkeypatch):
    """Sanity check alongside the fabrication tests above: the guard
    rejects fake quotes without also rejecting real ones."""

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        payload = {
            "theme": "Wants a Docker follow-up",
            "quotes": REAL_COMMENTS,
            "suggested_title": "Docker 101",
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    monkeypatch.setattr(insights, "acompletion", fake_acompletion)

    result = _run(insights.build_requests(_comments(), _stage_b(), _embeddings(), api_key=API_KEY))

    assert len(result) == 1
    assert set(result[0].quotes) == set(REAL_COMMENTS)
