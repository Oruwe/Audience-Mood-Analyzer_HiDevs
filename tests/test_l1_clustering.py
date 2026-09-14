"""L1 unit tests — engine.clustering.

Every other clustering test in this suite stubs the clusterer out, because
they are testing grouping and ordering logic rather than the maths. That
left the maths itself untested, which was survivable while it was
scikit-learn's and is not now that it is ours: this module replaced
`sklearn.cluster.KMeans` because importing scikit-learn cost ~180 MB on a
512 MiB instance and was OOM-killing the process mid-analysis.

So these test the properties the pipeline actually depends on, not an exact
label assignment — Lloyd's algorithm finds a local optimum, and which one
depends on seeding, so pinning specific labels would be testing the RNG.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.clustering import kmeans_labels


def _blobs(centres, per_blob=40, spread=0.25, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, spread, size=(per_blob, dim)) for c in centres])


def test_well_separated_groups_come_back_as_separate_clusters():
    """The one property everything downstream rests on: comments that are
    near-duplicates of each other land together, and distinct topics don't."""
    per_blob = 40
    matrix = _blobs([0.0, 6.0, 12.0], per_blob=per_blob)

    labels = kmeans_labels(matrix, 3, random_state=0)

    blocks = [labels[i * per_blob : (i + 1) * per_blob] for i in range(3)]
    for block in blocks:                      # each true group is unanimous
        assert len(set(block.tolist())) == 1
    assert len({block[0] for block in blocks}) == 3   # and they differ


def test_every_row_gets_exactly_one_label_in_range():
    matrix = _blobs([0.0, 5.0], per_blob=25)

    labels = kmeans_labels(matrix, 2, random_state=0)

    assert labels.shape == (50,)
    assert set(labels.tolist()) <= {0, 1}


def test_the_result_is_deterministic_for_a_given_seed():
    """`select_for_stage_b` takes a random_state and callers are entitled to
    assume the same corpus produces the same filter decisions twice."""
    matrix = _blobs([0.0, 4.0, 9.0], per_blob=20)

    first = kmeans_labels(matrix, 3, random_state=7)
    second = kmeans_labels(matrix, 3, random_state=7)

    assert np.array_equal(first, second)


def test_more_clusters_than_points_is_clamped_not_an_error():
    """k is derived from a comment count, so a tiny video legitimately asks
    for more clusters than it has comments. That should degrade, not raise."""
    matrix = _blobs([0.0], per_blob=3)

    labels = kmeans_labels(matrix, 10, random_state=0)

    assert labels.shape == (3,)
    assert len(set(labels.tolist())) <= 3


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_inputs_return_a_single_cluster(n):
    matrix = np.zeros((n, 4))

    labels = kmeans_labels(matrix, 3, random_state=0)

    assert labels.shape == (n,)
    assert labels.tolist() == [0] * n


def test_identical_points_do_not_hang_or_crash():
    """k-means++ divides by the total squared distance; when every point is
    the same that total is zero. Guarded, but worth pinning."""
    matrix = np.ones((20, 5))

    labels = kmeans_labels(matrix, 4, random_state=0)

    assert labels.shape == (20,)


def test_no_cluster_is_left_empty_when_k_points_are_distinct():
    """An empty cluster would silently give back fewer groups than asked
    for. The re-seeding branch exists to prevent that."""
    matrix = _blobs([0.0, 3.0, 6.0, 9.0], per_blob=10)

    labels = kmeans_labels(matrix, 4, random_state=0)

    assert len(set(labels.tolist())) == 4


def test_restarts_do_not_make_the_fit_worse():
    """n_init exists because a single run lands in a local optimum. More
    restarts must never increase inertia."""
    matrix = _blobs([0.0, 1.5, 3.0], per_blob=30, spread=0.6)

    def inertia(labels):
        total = 0.0
        for label in set(labels.tolist()):
            members = matrix[labels == label]
            total += float(np.sum((members - members.mean(axis=0)) ** 2))
        return total

    one = inertia(kmeans_labels(matrix, 3, n_init=1, random_state=0))
    many = inertia(kmeans_labels(matrix, 3, n_init=20, random_state=0))

    assert many <= one + 1e-9


def test_a_non_matrix_input_is_rejected_rather_than_silently_reshaped():
    with pytest.raises(ValueError):
        kmeans_labels(np.zeros(10), 2)
