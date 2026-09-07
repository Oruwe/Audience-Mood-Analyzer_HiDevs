"""Phase 4: statistical anomaly detection over the rolling urgency window."""

import logging
import uuid
from datetime import UTC, datetime

import numpy as np

from engine.topic_cluster import extract_trending_theme
from schemas import CrisisAlert
from storage.db import aget_comments_since

logger = logging.getLogger(__name__)

_MEAN_URGENCY_TRIGGER = 0.70
_SPIKE_URGENCY = 0.85
_MIN_SPIKE_COUNT = 3


async def detect_anomalies(window_minutes: int = 15) -> CrisisAlert | None:
    """Return a CrisisAlert when window urgency spikes, otherwise None."""
    try:
        comments = await aget_comments_since(window_minutes)
    except FileNotFoundError:
        return None  # warehouse not created yet — nothing to scan
    if not comments:
        return None

    urgencies = np.array([c.urgency_score for c in comments], dtype=float)
    mean_urgency = float(urgencies.mean())
    std_urgency = float(urgencies.std())
    spike_ids = [c.comment_id for c in comments if c.urgency_score > _SPIKE_URGENCY]

    mean_triggered = mean_urgency > _MEAN_URGENCY_TRIGGER
    spike_triggered = len(spike_ids) >= _MIN_SPIKE_COUNT
    if not (mean_triggered or spike_triggered):
        return None

    urgent = [c for c in comments if c.urgency_score > _SPIKE_URGENCY] or comments
    theme = await extract_trending_theme(urgent)

    severity = "CRITICAL" if mean_triggered else "WARNING"
    reason = (
        f"mean urgency {mean_urgency:.2f} (std {std_urgency:.2f}) across "
        f"{len(comments)} comments in last {window_minutes} min; "
        f"{len(spike_ids)} above {_SPIKE_URGENCY}"
    )
    logger.warning("anomaly detected: %s | %s", theme, reason)
    return CrisisAlert(
        alert_id=f"alert-{uuid.uuid4().hex[:12]}",
        timestamp=datetime.now(UTC),
        severity=severity,
        theme=theme,
        trigger_reason=reason,
        affected_comment_ids=[c.comment_id for c in comments],
    )
