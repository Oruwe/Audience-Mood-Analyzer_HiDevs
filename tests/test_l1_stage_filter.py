"""L1 unit tests — engine.stage_filter (SPEC §4.1 Stage A -> Stage B filter).

KMeans itself is stubbed out (fixed label arrays) so these tests are pure,
deterministic checks of *our* thresholding logic — strong-negative /
high-confidence / dense-cluster — not of scikit-learn's clustering quality,
which isn't what this module is responsible for getting right.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

import engine.stage_filter as stage_filter
from engine.stage_filter import _cluster_count, flag_comments, select_for_stage_b
from schemas import RawComment, Sentiment, StageASentimentItem


def _comment(i: int) -> RawComment:
    return RawComment(
        id=f"c{i}", platform="youtube", text=f"comment {i}",
        timestamp=datetime.now(timezone.utc), video_id="v1",
    )


def _sentiment(comment_id: str, sentiment: Sentiment, confidence: float) -> StageASentimentItem:
    return StageASentimentItem(comment_id=comment_id, sentiment=sentiment, confidence=confidence)


def _stub_kmeans(monkeypatch, labels: list[int]):
    """Replace KMeans with a stub that returns a fixed label array,
    regardless of k/n_init/random_state or the actual embedding values."""

    class _StubKMeans:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, matrix):
            assert len(matrix) == len(labels)
            return np.array(labels)

    monkeypatch.setattr(stage_filter, "KMeans", _StubKMeans)


def _forbid_kmeans(monkeypatch):
    def _boom(**kwargs):
        raise AssertionError("KMeans should not be instantiated for < 2 comments")

    monkeypatch.setattr(stage_filter, "KMeans", _boom)


@pytest.mark.parametrize("n,expected_k", [
    (0, 1),
    (1, 1),
    (2, 2),
    (10, 2),     # max(2, 10//15=0) = 2
    (30, 2),     # max(2, 30//15=2) = 2
    (100, 6),    # max(2, 100//15=6) = 6
    (2000, 60),  # max(2, 133) capped at MAX_CLUSTERS=60
])
def test_cluster_count_formula(n, expected_k):
    assert _cluster_count(n) == expected_k


def test_empty_input_returns_empty_without_touching_kmeans(monkeypatch):
    _forbid_kmeans(monkeypatch)
    assert flag_comments([], {}, {}) == {}
    assert select_for_stage_b([], {}, {}) == []


def test_single_comment_skips_clustering_entirely(monkeypatch):
    _forbid_kmeans(monkeypatch)
    comments = [_comment(0)]
    sentiments = {"c0": _sentiment("c0", Sentiment.NEUTRAL, 0.5)}
    embeddings = {"c0": [0.0, 0.0]}

    reasons = flag_comments(comments, sentiments, embeddings)

    assert reasons["c0"].dense_cluster is False
    assert reasons["c0"].flagged is False  # neutral, low confidence, alone


def test_strong_negative_flags_regardless_of_confidence(monkeypatch):
    _forbid_kmeans(monkeypatch)
    comments = [_comment(0)]
    sentiments = {"c0": _sentiment("c0", Sentiment.NEGATIVE, 0.1)}  # low confidence
    embeddings = {"c0": [0.0, 0.0]}

    reasons = flag_comments(comments, sentiments, embeddings)

    assert reasons["c0"].strong_negative is True
    assert reasons["c0"].flagged is True


def test_critical_escalation_counts_as_strong_negative(monkeypatch):
    _forbid_kmeans(monkeypatch)
    comments = [_comment(0)]
    sentiments = {"c0": _sentiment("c0", Sentiment.CRITICAL_ESCALATION, 0.2)}
    embeddings = {"c0": [0.0, 0.0]}

    assert flag_comments(comments, sentiments, embeddings)["c0"].flagged is True


def test_high_confidence_flags_regardless_of_positive_sentiment(monkeypatch):
    """SPEC lists 'high-confidence' separately from 'strong negative' —
    that only makes sense if it also catches confident *positive* comments
    (e.g. a confident, enthusiastic request). See module docstring."""
    _forbid_kmeans(monkeypatch)
    comments = [_comment(0)]
    sentiments = {"c0": _sentiment("c0", Sentiment.STRONGLY_POSITIVE, 0.95)}
    embeddings = {"c0": [0.0, 0.0]}

    reasons = flag_comments(comments, sentiments, embeddings)

    assert reasons["c0"].strong_negative is False
    assert reasons["c0"].high_confidence is True
    assert reasons["c0"].flagged is True


def test_low_confidence_neutral_singleton_clusters_are_not_flagged(monkeypatch):
    comments = [_comment(0), _comment(1)]
    sentiments = {
        "c0": _sentiment("c0", Sentiment.NEUTRAL, 0.5),
        "c1": _sentiment("c1", Sentiment.NEUTRAL, 0.5),
    }
    embeddings = {"c0": [0.0, 0.0], "c1": [9.0, 9.0]}
    _stub_kmeans(monkeypatch, labels=[0, 1])  # each its own cluster of size 1

    reasons = flag_comments(comments, sentiments, embeddings, dense_cluster_min_size=5)

    assert reasons["c0"].flagged is False
    assert reasons["c1"].flagged is False


def test_dense_cluster_flags_even_low_confidence_neutral_comments(monkeypatch):
    comments = [_comment(i) for i in range(6)]
    sentiments = {
        c.id: _sentiment(c.id, Sentiment.NEUTRAL, 0.3) for c in comments
    }
    embeddings = {c.id: [float(i), 0.0] for i, c in enumerate(comments)}
    # First 5 comments share cluster 0 (size 5, meets the default min of 5);
    # the last is alone in cluster 1 (size 1).
    _stub_kmeans(monkeypatch, labels=[0, 0, 0, 0, 0, 1])

    reasons = flag_comments(comments, sentiments, embeddings)

    for i in range(5):
        assert reasons[f"c{i}"].dense_cluster is True
        assert reasons[f"c{i}"].flagged is True
    assert reasons["c5"].dense_cluster is False
    assert reasons["c5"].flagged is False


def test_select_for_stage_b_preserves_original_order(monkeypatch):
    comments = [_comment(i) for i in range(4)]
    sentiments = {
        "c0": _sentiment("c0", Sentiment.NEUTRAL, 0.1),          # not flagged
        "c1": _sentiment("c1", Sentiment.NEGATIVE, 0.1),         # strong negative
        "c2": _sentiment("c2", Sentiment.NEUTRAL, 0.1),          # not flagged
        "c3": _sentiment("c3", Sentiment.STRONGLY_POSITIVE, 0.9),  # high confidence
    }
    embeddings = {c.id: [float(i), float(i)] for i, c in enumerate(comments)}
    _stub_kmeans(monkeypatch, labels=[0, 1, 2, 3])  # all singleton clusters -> no dense flags

    selected = select_for_stage_b(comments, sentiments, embeddings)

    assert [c.id for c in selected] == ["c1", "c3"]
