"""K-means, in numpy, because scikit-learn does not fit in 512 MiB.

This project used `sklearn.cluster.KMeans` in exactly two places, for
exactly one call each: `KMeans(...).fit_predict(matrix)` -> integer labels.
That single function cost **~180 MB of resident memory**, measured, on a
Render free instance whose entire budget is 512 MiB and which was already
carrying litellm (~195 MB), pandas (~89 MB) and streamlit (~31 MB). The
result was not slowness, it was an OOM kill mid-analysis: memory climbed to
536,768,500 bytes against a 536,870,900 limit and the process was restarted
twice, taking the background worker with it and leaving the page on
"Working…" forever (2026-09-14, 03:57:30 and 04:02:00).

So this is not "avoiding a dependency" as an aesthetic. Lloyd's algorithm
with k-means++ seeding is about fifty lines over numpy, which is already a
required dependency because the embeddings are numpy arrays anyway. Dropping
scikit-learn buys back a third of the instance.

What is deliberately NOT reimplemented: everything else scikit-learn's
KMeans offers -- elkan/triangle-inequality acceleration, sparse input,
sample weights, parallelism. None of it was used. The inputs here are a few
hundred to a few thousand dense 1536-dimensional vectors, where the plain
algorithm is comfortably fast.

Numerical note: this will not reproduce scikit-learn's labels for the same
seed, because the RNG draws differ. It reproduces its *behaviour* -- the
same objective, minimised over the same number of restarts. Cluster labels
were never stable across scikit-learn versions either; nothing downstream
depends on a specific label number, only on the grouping.
"""

from __future__ import annotations

import numpy as np

MAX_ITER = 100
TOLERANCE = 1e-6


def _kmeans_plusplus(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Seed k centres, each new one favoured by its distance from the rest.

    Plain random seeding regularly drops two centres into the same dense
    region and leaves a real cluster unrepresented; k-means++ is what makes
    a small `n_init` sufficient rather than a lottery.
    """
    n = matrix.shape[0]
    centres = np.empty((k, matrix.shape[1]), dtype=matrix.dtype)
    centres[0] = matrix[rng.integers(n)]

    # Squared distance from every point to the nearest centre chosen so far.
    closest_sq = np.sum((matrix - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:  # every point already coincides with a centre
            centres[i] = matrix[rng.integers(n)]
        else:
            centres[i] = matrix[rng.choice(n, p=closest_sq / total)]
        closest_sq = np.minimum(closest_sq, np.sum((matrix - centres[i]) ** 2, axis=1))
    return centres


def _assign(matrix: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, float]:
    """Label every point by its nearest centre, and report the inertia.

    ||x - c||^2 is expanded to ||x||^2 - 2x·c + ||c||^2 and the ||x||^2 term
    dropped: it is identical across centres, so it cannot change which one
    wins, and leaving it out avoids materialising an (n, k, dim) difference
    array that would dwarf the memory this module exists to save.
    """
    cross = matrix @ centres.T
    centre_sq = np.sum(centres ** 2, axis=1)
    partial = centre_sq[None, :] - 2.0 * cross
    labels = np.argmin(partial, axis=1)
    point_sq = np.sum(matrix ** 2, axis=1)
    inertia = float(np.sum(point_sq + partial[np.arange(matrix.shape[0]), labels]))
    return labels, inertia


def _lloyd(matrix: np.ndarray, k: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    centres = _kmeans_plusplus(matrix, k, rng)
    labels, inertia = _assign(matrix, centres)

    for _ in range(MAX_ITER):
        for j in range(k):
            members = matrix[labels == j]
            if members.size:
                centres[j] = members.mean(axis=0)
            else:
                # An empty cluster would stay empty forever. Re-seed it on the
                # point currently worst served by its own centre -- the same
                # repair scikit-learn makes, and the reason a degenerate k
                # does not silently collapse to fewer clusters than asked for.
                distances = np.sum((matrix - centres[labels]) ** 2, axis=1)
                centres[j] = matrix[int(np.argmax(distances))]

        new_labels, new_inertia = _assign(matrix, centres)
        converged = np.array_equal(new_labels, labels) or abs(inertia - new_inertia) <= TOLERANCE
        labels, inertia = new_labels, new_inertia
        if converged:
            break
    return labels, inertia


def kmeans_labels(matrix: np.ndarray, n_clusters: int, *, n_init: int = 10,
                  random_state: int = 0) -> np.ndarray:
    """Cluster *matrix* rows into *n_clusters*, returning one label per row.

    Drop-in for `KMeans(n_clusters=..., n_init=..., random_state=...)
    .fit_predict(matrix)`, which is the only way this project ever used
    scikit-learn. *n_init* restarts are run and the lowest-inertia one wins,
    because Lloyd's algorithm only finds a local optimum and which one it
    finds depends entirely on the seeding.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    n = matrix.shape[0]
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    # Asking for more clusters than points is not an error worth raising on:
    # callers derive k from a comment count, and a tiny channel should just
    # get one cluster per comment rather than a crash.
    k = min(n_clusters, n)
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=int)

    rng = np.random.default_rng(random_state)
    best_labels, best_inertia = None, np.inf
    for _ in range(max(1, n_init)):
        labels, inertia = _lloyd(matrix, k, rng)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels
