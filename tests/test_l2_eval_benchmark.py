"""L2 smoke test — evals.benchmark's scoring/metrics-writing logic.

Mocks engine.stage_a.classify_all_sentiments (no network, no OpenRouter
key) to prove the lenient 3-class scoring and data/eval_metrics.json
writing still work after Stage A moved from analyze_comment (V2's
Gemini/Groq router) to the batched OpenRouter classifier (SPEC.md §4.1
amendment) -- this file broke silently in that rewrite until fixed here.
"""

import asyncio
import json

import evals.benchmark as benchmark
from schemas import Sentiment, StageASentimentItem


def test_refuses_cleanly_without_an_openrouter_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        asyncio.run(benchmark.run_benchmark())
        raised = False
    except SystemExit as exc:
        raised = True
        assert "OPENROUTER_API_KEY" in str(exc)
    assert raised


def test_scoring_and_metrics_file_with_a_stubbed_classifier(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    metrics_path = tmp_path / "eval_metrics.json"
    monkeypatch.setattr(benchmark, "METRICS_PATH", metrics_path)

    cases = [
        {"text": "great!", "platform": "youtube", "expected_mood": "positive"},
        {"text": "terrible", "platform": "youtube", "expected_mood": "negative"},
        {"text": "meh", "platform": "youtube", "expected_mood": "neutral"},
        {"text": "torn on this one", "platform": "youtube", "expected_mood": "mixed"},
    ]
    monkeypatch.setattr(benchmark, "load_dataset", lambda: cases)

    # Predictions: 3/3 non-mixed correct, mixed resolves to a polarity.
    predicted_sentiments = [
        Sentiment.STRONGLY_POSITIVE,  # coarses to "positive" -> matches case 0
        Sentiment.NEGATIVE,           # matches case 1
        Sentiment.NEUTRAL,            # matches case 2
        Sentiment.POSITIVE,           # case 3 is "mixed" -> counts as lenient-correct
    ]

    async def fake_classify_all_sentiments(comments, *, api_key, model=None):
        return {
            benchmark._comment_id(i): StageASentimentItem(
                comment_id=benchmark._comment_id(i), sentiment=s, confidence=0.9
            )
            for i, s in enumerate(predicted_sentiments)
        }

    monkeypatch.setattr(benchmark, "classify_all_sentiments", fake_classify_all_sentiments)

    asyncio.run(benchmark.run_benchmark())

    written = json.loads(metrics_path.read_text())
    assert written["total_cases"] == 4
    assert written["failed_cases"] == 0
    assert written["accuracy"] == 1.0          # all 3 non-mixed cases correct
    assert written["accuracy_lenient"] == 1.0  # + the mixed case resolves to a polarity
    assert written["mixed_cases"] == 1
    assert written["mixed_covered"] == 1


def test_partial_failure_is_recorded_not_silently_dropped(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    metrics_path = tmp_path / "eval_metrics.json"
    monkeypatch.setattr(benchmark, "METRICS_PATH", metrics_path)

    cases = [{"text": "x", "platform": "youtube", "expected_mood": "positive"}]
    monkeypatch.setattr(benchmark, "load_dataset", lambda: cases)

    async def fake_classify_all_sentiments(comments, *, api_key, model=None):
        raise RuntimeError("all providers failed")

    monkeypatch.setattr(benchmark, "classify_all_sentiments", fake_classify_all_sentiments)

    asyncio.run(benchmark.run_benchmark())

    written = json.loads(metrics_path.read_text())
    assert written["accuracy"] is None
    assert written["failed_cases"] == 1
    assert "RuntimeError" in written["error"]
