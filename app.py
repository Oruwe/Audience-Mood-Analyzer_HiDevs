"""SPEC §5 + §8 — the single Streamlit entrypoint. One process, one command:

    streamlit run app.py

No FastAPI hop (SPEC §2's cut of server.py: "for one Streamlit app it is
ceremony"). The expensive work — pulling comments, classifying them,
synthesizing insights — runs in a background thread (orchestration.py),
never blocking this script; this file only starts that job, polls its
Postgres-backed progress (`st.status`, refreshed via `streamlit_autorefresh`
rather than a real blocking wait), and renders SPEC §3's three insight
blocks once it's done. SPEC §10 invariant 4 holds throughout: every
comment quote goes through plain `st.markdown` with Streamlit's default
HTML-escaping left untouched, so no raw markup or script from a comment
ever renders as anything but literal text.

Module layout: the functions above `main()` are plain, framework-free
logic (URL validation via ingestion.youtube, the SPEC §4.4 quota
pre-flight, the SPEC §4.2 cache-by-comment-count lookup, and Postgres
reads) — each independently unit-testable by importing this file as a
normal module, with no Streamlit runtime required. `main()` and the
`_render_*` functions below it are the Streamlit glue on top.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ingestion.youtube import (
    QuotaEstimate,
    QuotaExceededError,
    QuotaLedger,
    UnparsableURLError,
    YouTubeIngestionError,
    estimate_channel_analysis,
)
from orchestration import cancel_analysis, start_analysis
from schemas import ChannelInsights
from storage.postgres import (
    JobProgress,
    connect,
    find_reusable_job,
    get_batch_results,
    get_job_progress,
    init_schema,
)

REQUIRED_ENV_VARS = ("YOUTUBE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL")

_STAGE_LABELS = {
    "ingestion": "Pulling comments from YouTube",
    "stage_a_sentiment": "Classifying sentiment (Stage A)",
    "stage_a_embedding": "Embedding comments (Stage A)",
    "stage_b": "Classifying requests & confusion (Stage B)",
    "stage_c": "Synthesizing insights (Stage C)",
}

POLL_INTERVAL_MS = 3_000


# ---------------------------------------------------------------------------
# Logic layer — no Streamlit calls below this line, only above `main()`.
# ---------------------------------------------------------------------------

def missing_config() -> list[str]:
    return [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]


def stage_label(stage: str | None) -> str:
    if stage is None:
        return "Starting…"
    return _STAGE_LABELS.get(stage, stage)


def format_quota_refusal(estimate: QuotaEstimate, remaining: int) -> str:
    """SPEC §4.4: "If a channel analysis would exceed quota, say so in the
    UI up front. A clear refusal reads as competence; a spinner that dies
    reads as broken." """
    return (
        f"This analysis needs an estimated {estimate.units_required_for_comment_pull} "
        f"more YouTube API quota units to pull {estimate.total_comment_count} comments "
        f"across {len(estimate.videos)} videos, but only {remaining} remain today "
        "(resets at midnight Pacific Time)."
    )


async def preflight(
    channel_ref: str, *, youtube_api_key: str, ledger: QuotaLedger
) -> QuotaEstimate:
    """The cheap SPEC §4.4 estimate — resolves the channel/video and counts
    comments, without pulling a single one. Raises UnparsableURLError,
    ChannelNotFoundError, or QuotaExceededError (if even this cheap pass
    can't be afforded) — callers should let those surface as st.error.
    """
    async with httpx.AsyncClient() as client:
        return await estimate_channel_analysis(
            channel_ref, client=client, api_key=youtube_api_key, ledger=ledger,
        )


async def find_cached_analysis(channel_ref: str, total_comment_count: int) -> str | None:
    """SPEC §4.2: reuse a previous completed analysis of this exact channel
    at this exact comment count, instead of spending quota again."""
    async with connect() as conn:
        await init_schema(conn)
        return await find_reusable_job(conn, channel_ref, total_comment_count)


async def launch_analysis(
    channel_ref: str, *, youtube_api_key: str, openrouter_api_key: str
) -> str:
    return await start_analysis(
        channel_ref, youtube_api_key=youtube_api_key, openrouter_api_key=openrouter_api_key,
    )


async def load_progress(job_id: str) -> JobProgress | None:
    async with connect() as conn:
        return await get_job_progress(conn, job_id)


async def load_insights(job_id: str) -> ChannelInsights | None:
    async with connect() as conn:
        results = await get_batch_results(conn, job_id, "stage_c")
    raw = results.get("insights")
    return ChannelInsights.model_validate(raw) if raw else None


# ---------------------------------------------------------------------------
# Streamlit glue
# ---------------------------------------------------------------------------

@st.cache_resource
def _quota_ledger() -> QuotaLedger:
    """One process-wide ledger (SPEC §9: "Quota / rate counters: In-memory
    v1"), shared across every rerun and every browser session hitting this
    deployment — st.cache_resource is Streamlit's way to say "build this
    once per process," which is exactly the semantics an in-memory daily
    quota counter needs.
    """
    return QuotaLedger()


def _run_async(coro):
    return asyncio.run(coro)


def main() -> None:
    st.set_page_config(page_title="Audience Mood Analyzer", page_icon="🎥", layout="wide")
    st.title("🎥 Audience Mood Analyzer")
    st.caption(
        "Paste a YouTube channel or video URL — no signup. Get back what your "
        "audience is actually asking for, where your explanation didn't land, "
        "and which video landed badly, each backed by real comments."
    )

    missing = missing_config()
    if missing:
        st.error(
            "Missing configuration: " + ", ".join(missing) + ". "
            "Copy .env.example to .env, fill these in, and restart the app."
        )
        st.stop()

    channel_ref = st.text_input(
        "YouTube channel or video URL",
        placeholder="https://www.youtube.com/@channel",
        key="channel_ref_input",
    )
    analyze_clicked = st.button(
        "Analyze", type="primary", disabled=not channel_ref.strip()
    )

    if analyze_clicked:
        _start_new_analysis(channel_ref.strip())

    job_id = st.session_state.get("job_id")
    if job_id:
        _render_job(job_id)


def _start_new_analysis(channel_ref: str) -> None:
    youtube_key = os.environ["YOUTUBE_API_KEY"]
    openrouter_key = os.environ["OPENROUTER_API_KEY"]
    ledger = _quota_ledger()

    try:
        with st.spinner("Checking video count and quota…"):
            estimate = _run_async(
                preflight(channel_ref, youtube_api_key=youtube_key, ledger=ledger)
            )
    except UnparsableURLError as exc:
        st.error(str(exc))
        return
    except QuotaExceededError as exc:
        st.error(f"Can't run this analysis today: {exc}")
        return
    except YouTubeIngestionError as exc:
        st.error(f"Couldn't reach that channel/video: {exc}")
        return

    if estimate.units_required_for_comment_pull > ledger.remaining:
        st.error(format_quota_refusal(estimate, ledger.remaining))
        return

    cached_job_id = _run_async(
        find_cached_analysis(channel_ref, estimate.total_comment_count)
    )
    if cached_job_id:
        st.session_state["job_id"] = cached_job_id
        st.session_state["served_from_cache"] = True
        st.rerun()
        return

    job_id = _run_async(launch_analysis(
        channel_ref, youtube_api_key=youtube_key, openrouter_api_key=openrouter_key,
    ))
    st.session_state["job_id"] = job_id
    st.session_state["served_from_cache"] = False
    st.rerun()


def _render_job(job_id: str) -> None:
    progress = _run_async(load_progress(job_id))
    if progress is None:
        st.error("Lost track of that analysis — please start a new one.")
        return

    if progress.status in ("pending", "running"):
        _render_in_progress(job_id, progress)
        st_autorefresh(interval=POLL_INTERVAL_MS, key=f"poll_{job_id}")
        return

    if progress.status == "failed":
        st.error(f"Analysis failed: {progress.error}")
        return

    if progress.status == "cancelled":
        st.warning("Analysis cancelled.")
        return

    if st.session_state.get("served_from_cache"):
        st.success("Served from a previous analysis — this channel's comment count hasn't moved.")
    else:
        st.success("Analysis complete.")

    insights = _run_async(load_insights(job_id))
    if insights is None:
        st.error("Analysis completed but its insights are missing — this shouldn't happen.")
        return
    _render_insights(insights)


def _render_in_progress(job_id: str, progress: JobProgress) -> None:
    with st.status(stage_label(progress.stage), expanded=True):
        if progress.total_units:
            st.progress(
                progress.fraction_complete or 0.0,
                text=f"{progress.completed_units}/{progress.total_units}",
            )
        else:
            st.write("Working…")
        if st.button("Cancel analysis", key=f"cancel_{job_id}"):
            _run_async(cancel_analysis(job_id))
            st.warning("Cancellation requested — this may take a moment to stop.")


def _render_insights(insights: ChannelInsights) -> None:
    st.header("What your audience wants")
    if not insights.requests:
        st.caption("No clear content requests surfaced in this batch of comments.")
    for r in insights.requests:
        with st.container(border=True):
            st.subheader(r.theme)
            st.caption(f"Mentioned in {r.mention_count} comments")
            st.markdown(f"**Suggested title:** {r.suggested_title}")
            for quote in r.quotes:
                st.markdown(f"> {quote}")

    st.header("Where your explanation didn't land")
    if not insights.confusion_points:
        st.caption("No recurring confusion points surfaced in this batch of comments.")
    for c in insights.confusion_points:
        with st.container(border=True):
            st.subheader(c.sticking_point)
            caption = f"Mentioned in {c.mention_count} comments"
            if c.timestamp_hint:
                caption += f" · around {c.timestamp_hint}"
            st.caption(caption)
            for quote in c.quotes:
                st.markdown(f"> {quote}")

    st.header("Which video landed badly")
    if not insights.video_moods:
        st.caption("No video stood out as underperforming emotionally in this batch.")
    for v in insights.video_moods:
        with st.container(border=True):
            st.subheader(v.video_title)
            st.caption(
                f"Sentiment {v.sentiment_score:+.2f} "
                f"({v.delta_vs_channel_avg:+.2f} vs. channel average)"
            )
            st.markdown(f"**Likely driver:** {v.top_negative_driver}")
            for quote in v.quotes:
                st.markdown(f"> {quote}")


if __name__ == "__main__":
    main()
