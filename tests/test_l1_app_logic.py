"""L1 unit tests — app.py's pure logic helpers (no Streamlit runtime, no
network, no DB). `import app` is safe: every Streamlit call in the file
lives inside `main()`/the `_render_*` functions, guarded by
`if __name__ == "__main__"`, so importing the module as a normal Python
module never touches the Streamlit runtime.
"""

import app
from ingestion.youtube import ChannelInfo, QuotaEstimate, VideoMeta


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
