"""L1 unit tests — engine.insights's clustering helpers.

KMeans is stubbed (fixed label arrays), same approach as
tests/test_l1_stage_filter.py: these test *our* grouping/ordering/
size-filtering logic, not scikit-learn's clustering quality.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

import engine.insights as insights
from engine.insights import _cluster_comments, _cluster_count
from schemas import RawComment


def _comment(i: int) -> RawComment:
    return RawComment(
        id=f"c{i}", platform="youtube", text=f"comment {i}",
        timestamp=datetime.now(timezone.utc), video_id="v1",
    )


def _stub_kmeans(monkeypatch, labels: list[int]):
    class _StubKMeans:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, matrix):
            assert len(matrix) == len(labels)
            return np.array(labels)

    monkeypatch.setattr(insights, "KMeans", _StubKMeans)


@pytest.mark.parametrize("n,max_clusters,expected_k", [
    (0, 3, 1),
    (1, 3, 1),
    (2, 3, 1),
    (6, 3, 3),
    (100, 3, 3),
])
def test_cluster_count_formula(n, max_clusters, expected_k):
    assert _cluster_count(n, max_clusters) == expected_k


def test_fewer_than_two_comments_returns_no_clusters():
    assert _cluster_comments([], {}, max_clusters=3) == []
    one = [_comment(0)]
    assert _cluster_comments(one, {"c0": [0.0, 0.0]}, max_clusters=3) == []


def test_comments_missing_an_embedding_are_dropped_not_fatal(monkeypatch):
    comments = [_comment(i) for i in range(4)]
    embeddings = {"c0": [0.0, 0.0], "c1": [0.0, 0.0]}  # c2, c3 missing
    _stub_kmeans(monkeypatch, labels=[0, 0])  # only 2 usable comments

    clusters = _cluster_comments(comments, embeddings, max_clusters=3)

    assert len(clusters) == 1
    assert {c.id for c in clusters[0]} == {"c0", "c1"}


def test_clusters_smaller_than_min_size_are_dropped(monkeypatch):
    comments = [_comment(i) for i in range(5)]
    embeddings = {c.id: [float(i), 0.0] for i, c in enumerate(comments)}
    # cluster 0: c0,c1,c2 (size 3, kept); clusters 1 and 2 are singletons (dropped)
    _stub_kmeans(monkeypatch, labels=[0, 0, 0, 1, 2])

    clusters = _cluster_comments(comments, embeddings, max_clusters=3)

    assert len(clusters) == 1
    assert {c.id for c in clusters[0]} == {"c0", "c1", "c2"}


def test_clusters_are_ordered_biggest_first_and_capped(monkeypatch):
    comments = [_comment(i) for i in range(9)]
    embeddings = {c.id: [float(i), 0.0] for i, c in enumerate(comments)}
    # cluster sizes: 0->4, 1->3, 2->2
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2]
    _stub_kmeans(monkeypatch, labels=labels)

    clusters = _cluster_comments(comments, embeddings, max_clusters=2)

    assert len(clusters) == 2  # capped, even though 3 clusters exist
    assert [len(c) for c in clusters] == [4, 3]
