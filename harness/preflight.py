"""Preflight harness — prove every seam of this pipeline works, cheaply,
before a real analysis spends real money on a real channel.

Why this exists
---------------
Every production failure this project hit on its first live day was a
*seam* failure, not a logic failure, and every one of them was invisible
to the unit suite because the unit suite mocks the seam:

  - a model slug that litellm's registry listed but OpenRouter had
    withdrawn from the free tier (404 mid-analysis)
  - litellm's `openrouter/` routing prefix sent verbatim to OpenRouter's
    own REST embeddings endpoint ("model does not exist")
  - an embedding model whose output dimension was documented but never
    once verified against a live response
  - a provider returning a transient error shape the retry logic didn't
    recognise, so it neither retried nor fell back
  - a Postgres reachable from one network path and not another

Each of those cost a full failed run to discover. Each is a ten-second
check. That asymmetry is the whole argument for this file.

Design rules
------------
1. **Exercise production code paths, never parallel reimplementations.**
   The model probes call `engine.batching`/`engine.stage_a`/
   `engine.insights` exactly as the pipeline does, so a PASS here means
   *that code* works against *that provider* — not that some simplified
   lookalike does.
2. **Never abort on first failure.** Every check runs and reports, so one
   invocation tells you everything that's broken, not just the first
   thing.
3. **Spend as close to nothing as possible.** Probes use 2-3 synthetic
   comments. Real spend is measured, not estimated: OpenRouter's own
   `/api/v1/key` endpoint reports cumulative credit usage, so the report
   quotes the actual before/after delta.
4. **Be honest about what wasn't checked.** A SKIP is reported as loudly
   as a FAIL; a check that couldn't run is never quietly counted as
   passing.

Usage
-----
    python -m harness.preflight              # everything, incl. live probes
    python -m harness.preflight --offline    # config + Postgres only, $0
    python -m harness.preflight --json       # machine-readable

Also runnable from the deployed app (app.py's diagnostics panel), which is
usually the more useful place: it tests the network path that production
actually uses, rather than a developer laptop's.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

import config.models as models
from engine.batching import run_batched_llm_classification
from engine.insights import REQUEST_SYSTEM_PROMPT, _synthesize
from engine.llm_client import SYSTEM_PROMPT as STAGE_B_PROMPT
from engine.stage_a import SENTIMENT_SYSTEM_PROMPT, embed_comments_batch
from schemas import (
    RawComment,
    RequestInsightDraft,
    StageASentimentBatch,
    StageBClassificationBatch,
)
from storage.postgres import (
    connect,
    create_job,
    get_job_progress,
    init_schema,
    reap_stale_jobs,
    record_batch_result,
    touch_heartbeat,
)

OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# "Me at the zoo" — the first video on YouTube. A stable, public, never-
# going-away id to prove the API key works without depending on anything
# of the user's own.
_PROBE_VIDEO_ID = "jNQXAC9IVRw"

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in (PASS, WARN, SKIP)


@dataclass
class PreflightReport:
    results: list[CheckResult] = field(default_factory=list)
    credits_spent: float | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def failed(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def skipped(self) -> list[CheckResult]:
        return [r for r in self.results if r.status == SKIP]

    @property
    def ready(self) -> bool:
        """True only when nothing failed AND nothing was skipped.

        A skip is not a pass: it means that seam is still unproven, which
        is exactly the state this harness exists to make visible.
        """
        return not self.failed and not self.skipped


def _probe_comments(n: int = 2) -> list[RawComment]:
    """Synthetic comments with deliberately awkward ids.

    The ids are not `c0`/`c1`: a model that renumbers, truncates, or
    "tidies" ids is the exact failure the §4.1b guard catches in
    production, and a probe using trivial ids wouldn't provoke it.
    """
    texts = [
        "this finally made sense to me, thank you!!",
        "wait what happened at 4:32, mine throws an error there",
        "please do a part 2 on the deployment side",
    ]
    return [
        RawComment(
            id=f"probe-{i}-Zx{i}9", platform="youtube", text=texts[i % len(texts)],
            timestamp=datetime.now(timezone.utc), video_id="probe-video",
        )
        for i in range(n)
    ]


async def _timed(name: str, coro) -> CheckResult:
    """Run one check, turning any exception into a FAIL rather than
    letting it abort the rest of the run."""
    start = time.monotonic()
    try:
        status, detail = await coro
    except Exception as exc:  # noqa: BLE001 -- a check crashing IS a failure
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    return CheckResult(name, status, detail, time.monotonic() - start)


# ---------------------------------------------------------------------------
# Offline checks — no network, no spend
# ---------------------------------------------------------------------------

async def check_environment() -> tuple[str, str]:
    required = ("YOUTUBE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return FAIL, f"missing: {', '.join(missing)}"
    return PASS, f"all {len(required)} present"


async def check_model_config() -> tuple[str, str]:
    """Catch the config mistakes that have actually shipped here before.

    Both of these were real: a model string missing litellm's routing
    prefix (so it never routes to OpenRouter at all), and an embedding
    dimension constant left out of sync with the embedding model beside
    it (so clustering silently operates on wrong-shaped vectors).
    """
    chat = {
        "STAGE_A_SENTIMENT": models.STAGE_A_SENTIMENT,
        "STAGE_A_SENTIMENT_FALLBACK": models.STAGE_A_SENTIMENT_FALLBACK,
        "STAGE_B_CLASSIFY": models.STAGE_B_CLASSIFY,
        "STAGE_B_CLASSIFY_FALLBACK": models.STAGE_B_CLASSIFY_FALLBACK,
        "STAGE_C_SYNTHESIS": models.STAGE_C_SYNTHESIS,
        "STAGE_C_SYNTHESIS_FALLBACK": models.STAGE_C_SYNTHESIS_FALLBACK,
        "STAGE_A_EMBEDDINGS": models.STAGE_A_EMBEDDINGS,
        "STAGE_A_EMBEDDINGS_FALLBACK": models.STAGE_A_EMBEDDINGS_FALLBACK,
    }
    problems = [
        f"{name} is not an openrouter/* string ({value!r})"
        for name, value in chat.items()
        if not value.startswith("openrouter/")
    ]
    if not isinstance(models.EMBEDDING_DIM, int) or models.EMBEDDING_DIM <= 0:
        problems.append(f"EMBEDDING_DIM is not a positive int ({models.EMBEDDING_DIM!r})")

    # A stage sharing its primary with its own fallback isn't an error, but
    # it does mean that stage has no real insurance -- one provider outage
    # takes out both. Worth saying out loud. Embeddings is excluded on
    # purpose: config/models.py documents why its fallback must match
    # dimension, which made same-vendor the deliberate, correct choice
    # there rather than an oversight.
    same_vendor = [
        stage for stage, (primary, fb) in {
            "Stage A · sentiment": (models.STAGE_A_SENTIMENT, models.STAGE_A_SENTIMENT_FALLBACK),
            "Stage B": (models.STAGE_B_CLASSIFY, models.STAGE_B_CLASSIFY_FALLBACK),
            "Stage C": (models.STAGE_C_SYNTHESIS, models.STAGE_C_SYNTHESIS_FALLBACK),
        }.items()
        if primary.split("/")[1] == fb.split("/")[1]
    ]

    if problems:
        return FAIL, "; ".join(problems)
    if same_vendor:
        return WARN, f"{', '.join(same_vendor)} share a vendor with their fallback"
    return PASS, f"8 model strings well-formed, EMBEDDING_DIM={models.EMBEDDING_DIM}"


async def check_postgres() -> tuple[str, str]:
    """Full round-trip against the real database: schema, write, read back,
    heartbeat, reap. Anything the pipeline does to Postgres, this does."""
    async with connect() as conn:
        await init_schema(conn)
        job_id = await create_job(conn, "__preflight__")
        await record_batch_result(conn, job_id, "preflight", "0", result={"ok": True})
        await touch_heartbeat(conn, job_id)
        progress = await get_job_progress(conn, job_id)
        if progress is None:
            return FAIL, "wrote a job row but could not read it back"
        if progress.heartbeat_at is None:
            return FAIL, "heartbeat_at did not persist (is the migration applied?)"
        reaped = await reap_stale_jobs(conn)
        # Clean up after ourselves -- cascade removes the batch row too.
        await conn.execute("DELETE FROM analysis_jobs WHERE id = $1", job_id)
        return PASS, f"schema ok, round-trip ok, reaper ok ({reaped} stale job(s) cleaned)"


# ---------------------------------------------------------------------------
# Live checks — network, minimal spend
# ---------------------------------------------------------------------------

async def _openrouter_credits(client: httpx.AsyncClient, api_key: str) -> dict | None:
    response = await client.get(
        OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}
    )
    if response.status_code != 200:
        return None
    return response.json().get("data") or {}


async def check_openrouter_key(client: httpx.AsyncClient, api_key: str) -> tuple[str, str]:
    """Validate the key and its balance without spending a token.

    Matters because every stage is now on a paid model: a key with no
    credit fails *every* analysis, and it fails it deep inside Stage A
    rather than at the door.
    """
    data = await _openrouter_credits(client, api_key)
    if data is None:
        return FAIL, "key rejected by OpenRouter (401/403) — check OPENROUTER_API_KEY"
    usage = data.get("usage")
    limit = data.get("limit")
    if limit is None:
        return PASS, f"key valid, unlimited credit (used ${usage or 0:.4f} so far)"
    remaining = float(limit) - float(usage or 0)
    if remaining <= 0:
        return FAIL, f"key valid but OUT OF CREDIT (${remaining:.4f} remaining)"
    if remaining < 0.05:
        return WARN, f"key valid but low: ${remaining:.4f} remaining"
    return PASS, f"key valid, ${remaining:.4f} remaining"


async def check_youtube_key(client: httpx.AsyncClient, api_key: str) -> tuple[str, str]:
    """One quota unit against a known-stable public video."""
    response = await client.get(
        YOUTUBE_VIDEOS_URL,
        params={"part": "id,statistics", "id": _PROBE_VIDEO_ID, "key": api_key},
    )
    if response.status_code == 403:
        body = response.text[:300]
        hint = "quota exhausted" if "quota" in body.lower() else "key rejected or API not enabled"
        return FAIL, f"HTTP 403 — {hint}: {body}"
    if response.status_code != 200:
        return FAIL, f"HTTP {response.status_code}: {response.text[:300]}"
    items = response.json().get("items", [])
    if not items:
        return FAIL, "API reachable but returned no data for a known-good video id"
    return PASS, "key valid, quota available (1 unit spent)"


async def _probe_chat_model(
    model: str, *, api_key: str, system_prompt: str, schema, stage_label: str,
) -> tuple[str, str]:
    """Drive one chat model through the real production call path.

    Uses `run_batched_llm_classification` — the same function the pipeline
    uses — with no fallback configured, so a failure is attributable to
    *this* model rather than being silently papered over by its backup.
    That is the point: the fallbacks get probed as their own checks.
    """
    comments = _probe_comments(2)
    results = await run_batched_llm_classification(
        comments, api_key=api_key, model=model, system_prompt=system_prompt,
        response_schema=schema, stage_label=stage_label, fallback_models=(),
    )
    returned = set(results)
    expected = {c.id for c in comments}
    if returned != expected:
        return FAIL, f"id mismatch: expected {sorted(expected)}, got {sorted(returned)}"
    return PASS, f"{len(results)}/{len(comments)} labelled, ids echoed exactly"


async def check_stage_a_sentiment(model: str, api_key: str) -> tuple[str, str]:
    return await _probe_chat_model(
        model, api_key=api_key, system_prompt=SENTIMENT_SYSTEM_PROMPT,
        schema=StageASentimentBatch, stage_label="preflight Stage A",
    )


async def check_stage_b_classify(model: str, api_key: str) -> tuple[str, str]:
    return await _probe_chat_model(
        model, api_key=api_key, system_prompt=STAGE_B_PROMPT,
        schema=StageBClassificationBatch, stage_label="preflight Stage B",
    )


async def check_stage_c_synthesis(model: str, api_key: str) -> tuple[str, str]:
    """Stage C's real path, including the verbatim-quote validator.

    This is the only check that exercises SPEC §10 invariant 3 against a
    live model: `_synthesize` parses with the corpus in context, so a
    model that paraphrases its quotes fails validation here exactly as it
    would in production (where it silently costs the user an entire
    insight block).
    """
    comments = _probe_comments(3)
    draft = await _synthesize(
        comments, api_key=api_key, model=model,
        system_prompt=REQUEST_SYSTEM_PROMPT, draft_schema=RequestInsightDraft,
        fallback_models=(),
    )
    if draft is None:
        return FAIL, "returned nothing schema-valid (bad JSON, or quotes failed the verbatim check)"
    return PASS, f"valid draft, {len(draft.quotes)} verbatim quote(s) accepted: {draft.theme!r}"


async def check_embeddings(model: str, api_key: str, client: httpx.AsyncClient) -> tuple[str, str]:
    """The check that would have caught two separate shipped bugs.

    Verifies the slug resolves (the `openrouter/` prefix bug), that the
    endpoint answers, and — the part nothing else in this codebase has
    ever verified against a live response — that the returned vector width
    actually equals EMBEDDING_DIM. A silent mismatch there doesn't crash;
    it quietly makes every downstream cluster meaningless.
    """
    comments = _probe_comments(2)
    vectors = await embed_comments_batch(comments, client=client, api_key=api_key, model=model)
    if set(vectors) != {c.id for c in comments}:
        return FAIL, f"expected {len(comments)} vectors, got {len(vectors)}"
    widths = {len(v) for v in vectors.values()}
    if len(widths) != 1:
        return FAIL, f"inconsistent vector widths across one batch: {sorted(widths)}"
    width = widths.pop()
    if width != models.EMBEDDING_DIM:
        return FAIL, (
            f"dimension mismatch: live model returns {width}, "
            f"config.models.EMBEDDING_DIM says {models.EMBEDDING_DIM} — "
            f"clustering would operate on wrong-shaped vectors"
        )
    return PASS, f"{len(vectors)} vectors, width {width} matches EMBEDDING_DIM"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_preflight(*, offline: bool = False) -> PreflightReport:
    report = PreflightReport()
    report.results.append(await _timed("Environment variables", check_environment()))
    report.results.append(await _timed("Model configuration", check_model_config()))

    if os.environ.get("DATABASE_URL"):
        report.results.append(await _timed("Postgres round-trip", check_postgres()))
    else:
        report.results.append(CheckResult("Postgres round-trip", SKIP, "DATABASE_URL not set"))

    if offline:
        for name in (
            "OpenRouter key + credit", "YouTube API key",
            "Stage A · sentiment", "Stage A · sentiment fallback",
            "Stage A · embeddings", "Stage A · embeddings fallback",
            "Stage B · classify", "Stage B · classify fallback",
            "Stage C · synthesis", "Stage C · synthesis fallback",
        ):
            report.results.append(CheckResult(name, SKIP, "--offline"))
        return report

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    youtube_key = os.environ.get("YOUTUBE_API_KEY", "")

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)) as client:
        credits_before = await _openrouter_credits(client, openrouter_key) if openrouter_key else None

        if youtube_key:
            report.results.append(await _timed("YouTube API key", check_youtube_key(client, youtube_key)))
        else:
            report.results.append(CheckResult("YouTube API key", SKIP, "YOUTUBE_API_KEY not set"))

        if not openrouter_key:
            for name in (
                "OpenRouter key + credit",
                "Stage A · sentiment", "Stage A · sentiment fallback",
                "Stage A · embeddings", "Stage A · embeddings fallback",
                "Stage B · classify", "Stage B · classify fallback",
                "Stage C · synthesis", "Stage C · synthesis fallback",
            ):
                report.results.append(CheckResult(name, SKIP, "OPENROUTER_API_KEY not set"))
            return report

        report.results.append(
            await _timed("OpenRouter key + credit", check_openrouter_key(client, openrouter_key))
        )
        report.results.append(await _timed(
            "Stage A · sentiment", check_stage_a_sentiment(models.STAGE_A_SENTIMENT, openrouter_key)))
        report.results.append(await _timed(
            "Stage A · sentiment fallback",
            check_stage_a_sentiment(models.STAGE_A_SENTIMENT_FALLBACK, openrouter_key)))
        report.results.append(await _timed(
            "Stage A · embeddings",
            check_embeddings(models.STAGE_A_EMBEDDINGS, openrouter_key, client)))
        report.results.append(await _timed(
            "Stage A · embeddings fallback",
            check_embeddings(models.STAGE_A_EMBEDDINGS_FALLBACK, openrouter_key, client)))
        report.results.append(await _timed(
            "Stage B · classify", check_stage_b_classify(models.STAGE_B_CLASSIFY, openrouter_key)))
        report.results.append(await _timed(
            "Stage B · classify fallback",
            check_stage_b_classify(models.STAGE_B_CLASSIFY_FALLBACK, openrouter_key)))
        report.results.append(await _timed(
            "Stage C · synthesis", check_stage_c_synthesis(models.STAGE_C_SYNTHESIS, openrouter_key)))
        report.results.append(await _timed(
            "Stage C · synthesis fallback",
            check_stage_c_synthesis(models.STAGE_C_SYNTHESIS_FALLBACK, openrouter_key)))

        credits_after = await _openrouter_credits(client, openrouter_key)
        if credits_before and credits_after:
            before, after = credits_before.get("usage"), credits_after.get("usage")
            if before is not None and after is not None:
                report.credits_spent = float(after) - float(before)

    return report


def format_report(report: PreflightReport) -> str:
    icons = {PASS: "✅", FAIL: "❌", WARN: "⚠️ ", SKIP: "⏭️ "}
    width = max(len(r.name) for r in report.results) if report.results else 0
    lines = ["", "Preflight — Audience Mood Analyzer", "=" * 72]
    for r in report.results:
        lines.append(f"{icons.get(r.status, '  ')} {r.name.ljust(width)}  {r.detail}")
    lines.append("=" * 72)

    counts = {s: sum(1 for r in report.results if r.status == s) for s in (PASS, WARN, FAIL, SKIP)}
    lines.append(
        f"{counts[PASS]} passed · {counts[WARN]} warned · "
        f"{counts[FAIL]} failed · {counts[SKIP]} skipped"
    )
    if report.credits_spent is not None:
        lines.append(f"OpenRouter credit actually spent by this run: ${report.credits_spent:.6f}")
    if report.failed:
        lines.append("")
        lines.append("NOT READY — fix the failures above before running a real analysis.")
    elif report.skipped:
        lines.append("")
        lines.append(
            "PARTIAL — nothing failed, but skipped checks are unproven seams, "
            "not passing ones."
        )
    else:
        lines.append("")
        lines.append("READY — every seam verified against its real provider.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.preflight",
        description="Validate every external seam before spending money on a real analysis.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="config + Postgres only; make no network calls and spend nothing",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report = asyncio.run(run_preflight(offline=args.offline))

    if args.json:
        print(json.dumps({
            "ready": report.ready,
            "credits_spent": report.credits_spent,
            "results": [
                {"name": r.name, "status": r.status, "detail": r.detail, "seconds": round(r.seconds, 3)}
                for r in report.results
            ],
        }, indent=2))
    else:
        print(format_report(report))

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
