"""L1 unit tests — app.py's pure logic helpers (no Streamlit runtime, no
network, no DB). `import app` is safe: every Streamlit call in the file
lives inside `main()`/the `_render_*` functions, guarded by
`if __name__ == "__main__"`, so importing the module as a normal Python
module never touches the Streamlit runtime.
"""

from datetime import datetime, timedelta, timezone

import pytest

import app
from ingestion.youtube import ChannelInfo, QuotaEstimate, VideoMeta
from orchestration import STAGE_CLASSIFY, STAGE_EMBEDDING, STAGE_INGESTION, STAGE_INSIGHTS, STAGE_SENTIMENT
from storage.postgres import JobProgress


def _estimate(total_comments: int, n_videos: int, pull_cost: int) -> QuotaEstimate:
    return QuotaEstimate(
        channel=ChannelInfo(channel_id="UCabc", title="Test Channel",
                             uploads_playlist_id="UUabc", video_count=n_videos),
        videos=[VideoMeta(video_id=f"v{i}", title=f"Video {i}", comment_count=1)
                for i in range(n_videos)],
        total_comment_count=total_comments,
        units_already_spent_on_estimate=1,
        units_required_for_comment_pull=pull_cost,
    )


def test_missing_config_lists_every_unset_var(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert set(app.missing_config()) == set(app.REQUIRED_ENV_VARS)


def test_missing_config_empty_once_everything_is_set(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "x")
    assert app.missing_config() == []


def test_missing_config_reports_only_the_actually_missing_ones(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "x")
    assert app.missing_config() == ["OPENROUTER_API_KEY"]


def test_stage_label_covers_every_orchestration_stage():
    import orchestration
    for stage in (
        orchestration.STAGE_INGESTION, orchestration.STAGE_SENTIMENT,
        orchestration.STAGE_EMBEDDING, orchestration.STAGE_CLASSIFY,
        orchestration.STAGE_INSIGHTS,
    ):
        label = app.stage_label(stage)
        assert label and label != stage  # every real stage gets a human label


def test_stage_label_handles_none_and_unknown_stage():
    assert app.stage_label(None) == "Starting…"
    assert app.stage_label("some_future_stage") == "some_future_stage"  # degrades, doesn't crash


def test_format_quota_refusal_includes_the_actual_numbers():
    estimate = _estimate(total_comments=5000, n_videos=42, pull_cost=120)
    message = app.format_quota_refusal(estimate, remaining=50)
    assert "120" in message
    assert "5000" in message
    assert "42" in message
    assert "50" in message
    assert "midnight Pacific" in message


# ---------------------------------------------------------------------------
# efficiency_summary / stage_durations_seconds / count_sentiments — the
# "Real-Time Efficiency" + "Visualization" evaluation-criteria helpers.
# ---------------------------------------------------------------------------

def _progress(**overrides) -> JobProgress:
    base = dict(
        job_id="j1", channel_ref="c", status="completed", stage=None,
        total_units=None, completed_units=0, total_comment_count=100,
        cancel_requested=False, error=None,
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc) + timedelta(seconds=10),
    )
    base.update(overrides)
    return JobProgress(**base)


def test_efficiency_summary_reports_rate_for_a_completed_job():
    progress = _progress(total_comment_count=200, started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                          finished_at=datetime(2026, 1, 1, 0, 0, 20, tzinfo=timezone.utc))
    summary = app.efficiency_summary(progress)
    assert summary is not None
    assert "200 comments" in summary
    assert "20.0s" in summary
    assert "10.0 comments/sec" in summary


def test_efficiency_summary_none_without_start_or_finish_timestamps():
    assert app.efficiency_summary(_progress(started_at=None)) is None
    assert app.efficiency_summary(_progress(finished_at=None)) is None


def test_efficiency_summary_none_without_a_comment_count():
    assert app.efficiency_summary(_progress(total_comment_count=None)) is None
    assert app.efficiency_summary(_progress(total_comment_count=0)) is None


def test_stage_durations_seconds_differences_consecutive_stage_checkpoints():
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    progress = _progress(started_at=start)
    summary = {
        STAGE_INGESTION: {"last_at": start + timedelta(seconds=5), "n": 2},
        STAGE_SENTIMENT: {"last_at": start + timedelta(seconds=12), "n": 3},
        STAGE_EMBEDDING: {"last_at": start + timedelta(seconds=15), "n": 3},
        STAGE_CLASSIFY: {"last_at": start + timedelta(seconds=18), "n": 1},
        STAGE_INSIGHTS: {"last_at": start + timedelta(seconds=19), "n": 1},  # single checkpoint
    }
    durations = app.stage_durations_seconds(progress, summary)
    assert durations[STAGE_INGESTION] == pytest.approx(5.0)
    assert durations[STAGE_SENTIMENT] == pytest.approx(7.0)
    assert durations[STAGE_EMBEDDING] == pytest.approx(3.0)
    assert durations[STAGE_CLASSIFY] == pytest.approx(3.0)
    # Stage C is checkpointed as a single unit -- a naive max-minus-min over
    # one timestamp would give 0; differencing against the previous stage's
    # checkpoint still yields its real, non-zero duration.
    assert durations[STAGE_INSIGHTS] == pytest.approx(1.0)


def test_stage_durations_seconds_skips_stages_with_no_checkpoints_yet():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    progress = _progress(started_at=start)
    summary = {STAGE_INGESTION: {"last_at": start + timedelta(seconds=2), "n": 1}}
    durations = app.stage_durations_seconds(progress, summary)
    assert set(durations) == {STAGE_INGESTION}


def test_stage_durations_seconds_empty_without_a_started_at():
    assert app.stage_durations_seconds(_progress(started_at=None), {}) == {}


def test_count_sentiments_aggregates_across_checkpointed_batches():
    batches = {
        "0": {"items": [
            {"comment_id": "c0", "sentiment": "positive"},
            {"comment_id": "c1", "sentiment": "negative"},
        ]},
        "2": {"items": [
            {"comment_id": "c2", "sentiment": "positive"},
            {"comment_id": "c3", "sentiment": "neutral"},
        ]},
    }
    assert app.count_sentiments(batches) == {"positive": 2, "negative": 1, "neutral": 1}


def test_count_sentiments_empty_for_no_batches():
    assert app.count_sentiments({}) == {}


# ---------------------------------------------------------------------------
# Chart builders — pure functions returning Altair chart objects; checked
# for the data they carry, not rendered (no browser in this test suite).
# ---------------------------------------------------------------------------

def test_confusion_matrix_chart_carries_every_cell_once():
    cm = [[3, 1, 0], [0, 4, 1], [1, 0, 5]]
    labels = ["positive", "neutral", "negative"]
    chart = app.confusion_matrix_chart(cm, labels)
    # A LayerChart (heatmap + count labels), both layers sharing one 9-row
    # long-form frame carried on the shared base chart.
    data = chart.data
    assert len(data) == 9
    assert set(data["actual"]) == set(labels)
    assert set(data["predicted"]) == set(labels)
    assert data["count"].sum() == sum(sum(row) for row in cm)


def test_sentiment_distribution_chart_includes_every_class_even_at_zero():
    chart = app.sentiment_distribution_chart({"positive": 5})
    assert len(chart.data) == len(app._SENTIMENT_ORDER)
    assert chart.data.set_index("sentiment").loc["positive", "count"] == 5
    assert chart.data.set_index("sentiment").loc["neutral", "count"] == 0


def test_stage_duration_chart_carries_one_row_per_stage():
    durations = {STAGE_INGESTION: 5.0, STAGE_SENTIMENT: 7.0}
    chart = app.stage_duration_chart(durations)
    assert len(chart.data) == 2
    assert set(chart.data["seconds"]) == {5.0, 7.0}
