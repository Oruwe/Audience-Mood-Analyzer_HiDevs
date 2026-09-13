"""L2 contract tests — engine.llm_client Stage B classification.

Mirrors tests/test_l2_stage_a_sentiment.py's structure since both stages
share engine.batching's §4.1b guard — stubs litellm.acompletion (patched on
engine.batching, where the call actually happens) with no network, no key.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import engine.batching as batching
import engine.llm_client as llm_client
from schemas import RawComment

API_KEY = "fake-openrouter-key"


def _comment(i: int) -> RawComment:
    return RawComment(
        id=f"c{i}", platform="youtube", text=f"comment number {i}",
        timestamp=datetime.now(timezone.utc), video_id="v1",
    )


def _fake_response(json_text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json_text))])


def _valid_batch_json(comments: list[RawComment]) -> str:
    return json.dumps({
        "results": [
            {"comment_id": c.id, "intent": "request", "is_request": True, "is_confusion": False}
            for c in comments
        ]
    })


def test_empty_batch_short_circuits_without_calling_the_model(monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("should not call acompletion for an empty batch")

    monkeypatch.setattr(batching, "acompletion", fail_if_called)
    result = asyncio.run(llm_client.classify_batch([], api_key=API_KEY))
    assert result == {}


def test_happy_path_returns_one_result_per_comment(monkeypatch):
    comments = [_comment(i) for i in range(5)]
    calls = {"count": 0}

    async def fake_acompletion(**kwargs):
        calls["count"] += 1
        return _fake_response(_valid_batch_json(comments))

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(llm_client.classify_batch(comments, api_key=API_KEY))

    assert calls["count"] == 1
    assert set(result.keys()) == {c.id for c in comments}
    assert all(item.is_request for item in result.values())
    assert all(item.intent.value == "request" for item in result.values())


def test_length_mismatch_splits_batch_and_retries(monkeypatch):
    comments = [_comment(i) for i in range(4)]
    calls: list[list[str]] = []

    async def fake_acompletion(**kwargs):
        content = kwargs["messages"][1]["content"]
        requested_ids = [line.split(":")[0] for line in content.splitlines()]
        calls.append(requested_ids)
        if len(requested_ids) == 4:
            return _fake_response(json.dumps({
                "results": [{"comment_id": requested_ids[0], "intent": "other",
                             "is_request": False, "is_confusion": False}]
            }))
        matching = [c for c in comments if c.id in requested_ids]
        return _fake_response(_valid_batch_json(matching))

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(llm_client.classify_batch(comments, api_key=API_KEY))

    assert set(result.keys()) == {c.id for c in comments}
    assert len(calls) == 3


def test_id_set_mismatch_raises_after_splitting_all_the_way_down(monkeypatch):
    comments = [_comment(i) for i in range(2)]

    async def fake_acompletion(**kwargs):
        return _fake_response(json.dumps({
            "results": [
                {"comment_id": "totally-unrelated-1", "intent": "other",
                 "is_request": False, "is_confusion": False},
                {"comment_id": "totally-unrelated-2", "intent": "other",
                 "is_request": False, "is_confusion": False},
            ]
        }))

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    with pytest.raises(llm_client.ClassificationBatchFailedError):
        asyncio.run(llm_client.classify_batch(comments, api_key=API_KEY))


def test_classify_all_batches_at_the_configured_size(monkeypatch):
    comments = [_comment(i) for i in range(10)]
    seen_batch_sizes: list[int] = []

    async def fake_acompletion(**kwargs):
        content = kwargs["messages"][1]["content"]
        n = len(content.splitlines())
        seen_batch_sizes.append(n)
        batch = [c for c in comments if c.id in content]
        return _fake_response(_valid_batch_json(batch))

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)
    result = asyncio.run(llm_client.classify_all(comments, api_key=API_KEY, batch_size=4))

    assert set(result.keys()) == {c.id for c in comments}
    assert seen_batch_sizes == [4, 4, 2]


def test_default_batch_size_is_within_spec_range():
    # SPEC §4.1: "40-60 comments per call"
    assert 40 <= llm_client.DEFAULT_BATCH_SIZE <= 60
