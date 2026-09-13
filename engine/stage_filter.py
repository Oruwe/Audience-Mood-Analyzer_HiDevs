"""SPEC §4.1: the Stage A -> Stage B filter.

"Stage B — generative, batched, on the filtered subset only (~10-20% of
comments). Only what Stage A flags: strong negative, high-confidence, or
inside a dense cluster."

Three independent criteria — a comment needs only one to be promoted to
Stage B:

1. **Strong negative** Stage A sentiment (NEGATIVE or CRITICAL_ESCALATION)
   — SPEC §3 Block 2 "confusion points" candidates.
2. **High-confidence** Stage A sentiment, regardless of polarity. SPEC
   lists this separately from "strong negative", which only makes sense if
   it also catches confident *positive* comments: "LOVED this, please make
   a Docker follow-up!" is high-confidence positive and is exactly a Block
   1 request. Filtering on negative sentiment alone would silently starve
   Block 1 of everything.
3. **Inside a dense cluster** — comments that closely resemble many others
   by embedding, even if each one individually reads as neutral or
   low-confidence, because "twenty people asked the same thing" is
   precisely what Blocks 1/2 need to surface, and no single one of those
   comments need look remarkable alone.

SPEC.md names the three criteria but not their thresholds — the constants
below are a documented, tunable starting point (SPEC §12: "extensible by
you"), not a measured optimum. Revisit once real channel data is available
(SPEC §11 Track 3, insight-quality eval).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from sklearn.cluster import KMeans

from schemas import RawComment, Sentiment, StageASentimentItem

HIGH_CONFIDENCE_THRESHOLD = 0.85
STRONG_NEGATIVE = frozenset({Sentiment.NEGATIVE, Sentiment.CRITICAL_ESCALATION})

# Dense-cluster detection deliberately uses a much finer k than Stage C's
# "capped at 8" synthesis clusters (SPEC §4.1 Stage C) -- this is catching
# small pockets of near-duplicate comments, not the ~8 broad themes Stage C
# reduces the whole channel to later. Roughly one cluster per 15 comments,
# bounded so a tiny or huge channel doesn't produce a degenerate k.
TARGET_CLUSTER_SIZE = 15
MIN_CLUSTERS = 2
MAX_CLUSTERS = 60
DENSE_CLUSTER_MIN_SIZE = 5


@dataclass(frozen=True)
class FlagReason:
    strong_negative: bool
    high_confidence: bool
    dense_cluster: bool

    @property
    def flagged(self) -> bool:
        return self.strong_negative or self.high_confidence or self.dense_cluster


def _cluster_count(n: int) -> int:
    if n < 2:
        return 1  # can't meaningfully cluster 0 or 1 points
    ideal = max(MIN_CLUSTERS, n // TARGET_CLUSTER_SIZE)
    return min(ideal, MAX_CLUSTERS, n)


def flag_comments(
    comments: list[RawComment],
    sentiments: dict[str, StageASentimentItem],
    embeddings: dict[str, list[float]],
    *,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
    dense_cluster_min_size: int = DENSE_CLUSTER_MIN_SIZE,
    random_state: int = 0,
) -> dict[str, FlagReason]:
    """Decide which of *comments* SPEC §4.1 says Stage B should see.

    *sentiments* and *embeddings* are Stage A's output, keyed by
    comment_id (engine.stage_a.classify_all_sentiments / embed_all_comments)
    — every comment in *comments* must have an entry in both.
    """
    reasons: dict[str, FlagReason] = {}
    for comment in comments:
        item = sentiments[comment.id]
        reasons[comment.id] = FlagReason(
            strong_negative=item.sentiment in STRONG_NEGATIVE,
            high_confidence=item.confidence >= high_confidence_threshold,
            dense_cluster=False,  # filled in below, once cluster membership is known
        )

    if len(comments) >= 2:
        ids = [c.id for c in comments]
        matrix = np.array([embeddings[i] for i in ids], dtype=float)
        k = _cluster_count(len(comments))
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(matrix)
        cluster_sizes = np.bincount(labels)
        for comment_id, label in zip(ids, labels):
            if cluster_sizes[label] >= dense_cluster_min_size:
                reasons[comment_id] = replace(reasons[comment_id], dense_cluster=True)

    return reasons


def select_for_stage_b(
    comments: list[RawComment],
    sentiments: dict[str, StageASentimentItem],
    embeddings: dict[str, list[float]],
    **kwargs,
) -> list[RawComment]:
    """Just the comments `flag_comments` flags, in their original order."""
    reasons = flag_comments(comments, sentiments, embeddings, **kwargs)
    return [c for c in comments if reasons[c.id].flagged]
