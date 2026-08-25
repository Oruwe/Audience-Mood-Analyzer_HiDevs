"""Phase 4: distil urgent comments into a short trending-theme label."""

import logging
import os

import litellm
import numpy as np
from sklearn.cluster import KMeans

from schemas import EnrichedCommentRecord

logger = logging.getLogger(__name__)

_THEME_MODEL = "gemini/gemini-2.5-flash"
_MIN_CLUSTER_SIZE = 3   # below this, skip clustering entirely
_TOP_K = 5              # summaries fed to the LLM
_FALLBACK_THEME = "General Feedback"


async def extract_trending_theme(comments: list[EnrichedCommentRecord]) -> str:
    """Return a 3-5 word label for the densest cluster of urgent comments."""
    if len(comments) < _MIN_CLUSTER_SIZE:
        return _FALLBACK_THEME

    embedded = [c for c in comments if c.embedding is not None]
    if len(embedded) < _MIN_CLUSTER_SIZE:
        # No usable vectors — fall back to the most urgent summaries.
        top = sorted(comments, key=lambda c: c.urgency_score, reverse=True)[:_TOP_K]
        summaries = [c.summary for c in top]
    else:
        matrix = np.array([c.embedding for c in embedded], dtype=float)
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=42)
        kmeans.fit(matrix)
        centroid = kmeans.cluster_centers_[0]
        distances = np.linalg.norm(matrix - centroid, axis=1)
        closest = np.argsort(distances)[:_TOP_K]
        summaries = [embedded[i].summary for i in closest]

    prompt = (
        "These are summaries of recent social-media comments about one emerging issue:\n"
        + "\n".join(f"- {s}" for s in summaries)
        + "\n\nSummarize this emerging issue in strictly 3 to 5 words."
    )
    try:
        response = await litellm.acompletion(
            model=_THEME_MODEL,
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            timeout=15,
        )
        theme = (response.choices[0].message.content or "").strip().strip('"')
        return theme or _FALLBACK_THEME
    except Exception as exc:  # noqa: BLE001 — radar must never crash the pipeline
        logger.warning("theme LLM call failed (%s); using fallback", exc)
        return _FALLBACK_THEME
