"""L1/L2 tests — harness/preflight.py.

The harness exists to catch seam failures that mocked unit tests can't, so
these tests do the one thing that is genuinely testable offline: prove the
*detectors* fire. Each test feeds a probe the exact shape of a real bug
this project shipped, and asserts the check reports FAIL rather than
shrugging it off — a diagnostic that passes on a broken system is worse
than no diagnostic, because it converts an outage into a mystery.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import config.models as models
import engine.batching as batching
from harness import preflight
from harness.preflight import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    CheckResult,
    PreflightReport,
    check_embeddings,
    check_environment,
    check_model_config,
    check_openrouter_key,
    check_postgres,
    check_stage_a_sentiment,
    check_youtube_key,
    format_report,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Offline checks
# ---------------------------------------------------------------------------

def test_environment_check_names_every_missing_variable(monkeypatch):
    for name in ("YOUTUBE_API_KEY", "OPENROUTER_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    status, detail = _run(check_environment())

    assert status == FAIL
    # All three named at once -- fixing them one restart at a time is the
    # slow path this harness exists to avoid.
    assert "YOUTUBE_API_KEY" in detail
    assert "OPENROUTER_API_KEY" in detail
    assert "DATABASE_URL" in detail


def test_environment_check_passes_when_all_present(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "x")

    assert _run(check_environment())[0] == PASS


def test_model_config_check_catches_a_missing_routing_prefix(monkeypatch):
    """The exact bug that shipped once: a model string without litellm's
    `openrouter/` prefix never routes to OpenRouter at all."""
    monkeypatch.setattr(models, "STAGE_B_CLASSIFY", "somevendor/model-without-prefix")

    status, detail = _run(check_model_config())

    assert status == FAIL
    assert "STAGE_B_CLASSIFY" in detail


def test_model_config_check_catches_a_nonsense_embedding_dim(monkeypatch):
    monkeypatch.setattr(models, "EMBEDDING_DIM", 0)

    status, detail = _run(check_model_config())

    assert status == FAIL
    assert "EMBEDDING_DIM" in detail


def test_model_config_check_warns_when_a_stage_shares_a_vendor_with_its_fallback(monkeypatch):
    """Not an error, but that stage has no real insurance: one provider
    outage takes out primary and backup together."""
    # Placeholder vendor/model names, not real slugs: SPEC §11.1 keeps every
    # real model string inside config/models.py, and this test only cares
    # about the primary and fallback sharing a vendor.
    monkeypatch.setattr(models, "STAGE_B_CLASSIFY", "openrouter/samevendor/model-a")
    monkeypatch.setattr(models, "STAGE_B_CLASSIFY_FALLBACK", "openrouter/samevendor/model-b")

    status, detail = _run(check_model_config())

    assert status == WARN
    assert "Stage B" in detail


def test_model_config_check_passes_on_the_real_shipped_config():
    assert _run(check_model_config())[0] in (PASS, WARN)


def test_postgres_check_round_trips_against_a_real_database(pg_dsn, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)

    status, detail = _run(check_postgres())

    assert status == PASS
    assert "round-trip ok" in detail


# ---------------------------------------------------------------------------
# Live-probe detectors, driven by fake transports/providers
# ---------------------------------------------------------------------------

def test_openrouter_check_fails_a_rejected_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "no"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_openrouter_key(client, "bad-key")

    status, detail = _run(scenario())
    assert status == FAIL
    assert "rejected" in detail


def test_openrouter_check_fails_an_exhausted_balance():
    """Every stage is on a paid model now, so a spent key fails every
    analysis -- and fails it deep inside Stage A rather than at the door."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"usage": 5.0, "limit": 5.0}})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_openrouter_key(client, "spent-key")

    status, detail = _run(scenario())
    assert status == FAIL
    assert "OUT OF CREDIT" in detail


def test_openrouter_check_warns_on_a_nearly_empty_balance():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"usage": 4.99, "limit": 5.0}})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_openrouter_key(client, "low-key")

    assert _run(scenario())[0] == WARN


def test_youtube_check_distinguishes_quota_exhaustion_from_a_bad_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text='{"error": {"message": "quotaExceeded"}}')

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_youtube_key(client, "key")

    status, detail = _run(scenario())
    assert status == FAIL
    assert "quota exhausted" in detail


def test_embeddings_check_catches_a_dimension_mismatch():
    """The check that nothing else in this codebase performs.

    EMBEDDING_DIM has only ever been documented, never verified against a
    live response. A model returning a different width doesn't crash --
    it silently makes every downstream cluster meaningless.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        wrong_width = [0.1] * (models.EMBEDDING_DIM - 1)
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": wrong_width},
            {"index": 1, "embedding": wrong_width},
        ]})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_embeddings(models.STAGE_A_EMBEDDINGS, "key", client)

    status, detail = _run(scenario())
    assert status == FAIL
    assert "dimension mismatch" in detail
    assert str(models.EMBEDDING_DIM) in detail


def test_embeddings_check_passes_on_the_declared_width():
    def handler(request: httpx.Request) -> httpx.Response:
        right_width = [0.1] * models.EMBEDDING_DIM
        return httpx.Response(200, json={"data": [
            {"index": 0, "embedding": right_width},
            {"index": 1, "embedding": right_width},
        ]})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await check_embeddings(models.STAGE_A_EMBEDDINGS, "key", client)

    status, detail = _run(scenario())
    assert status == PASS
    assert str(models.EMBEDDING_DIM) in detail


def test_chat_probe_fails_when_a_model_does_not_echo_ids_exactly(monkeypatch):
    """A model that renumbers ids passes JSON-schema validation and still
    corrupts the pipeline, because results are matched by comment_id."""
    async def renumbering_acompletion(*, model, **kwargs):
        payload = json.dumps({"results": [
            {"comment_id": "0", "sentiment": "positive", "confidence": 0.9},
            {"comment_id": "1", "sentiment": "neutral", "confidence": 0.8},
        ]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    monkeypatch.setattr(batching, "acompletion", renumbering_acompletion)

    # Through `_timed`, the way run_preflight always invokes a probe: the
    # §4.1b guard splits to single-item batches and then raises, and the
    # wrapper is what turns that into a reportable FAIL instead of an
    # abort that hides every other check.
    result = _run(preflight._timed(
        "Stage A · sentiment", check_stage_a_sentiment("openrouter/probe", "key")
    ))

    assert result.status == FAIL
    assert "classification failed" in result.detail


def test_chat_probe_passes_when_ids_are_echoed_exactly(monkeypatch):
    async def faithful_acompletion(*, model, messages, **kwargs):
        ids = [c["comment_id"] for c in json.loads(messages[-1]["content"])]
        payload = json.dumps({"results": [
            {"comment_id": cid, "sentiment": "positive", "confidence": 0.9} for cid in ids
        ]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    monkeypatch.setattr(batching, "acompletion", faithful_acompletion)

    status, detail = _run(check_stage_a_sentiment("openrouter/probe", "key"))

    assert status == PASS
    assert "echoed exactly" in detail


def test_a_crashing_check_is_reported_as_a_failure_not_an_abort():
    """One broken seam must never hide the state of the others."""
    async def boom():
        raise RuntimeError("provider exploded")

    result = _run(preflight._timed("exploding check", boom()))

    assert result.status == FAIL
    assert "provider exploded" in result.detail


# ---------------------------------------------------------------------------
# Report semantics
# ---------------------------------------------------------------------------

def test_a_skipped_check_is_not_counted_as_ready():
    """A skip means that seam is unproven. Treating it as a pass is how a
    harness lies to you."""
    report = PreflightReport(results=[
        CheckResult("a", PASS), CheckResult("b", SKIP, "no key"),
    ])

    assert report.failed == []
    assert report.ready is False


def test_report_is_ready_only_when_everything_actually_ran():
    report = PreflightReport(results=[CheckResult("a", PASS), CheckResult("b", WARN)])
    assert report.ready is True


def test_format_report_states_the_verdict_and_the_real_spend():
    report = PreflightReport(
        results=[CheckResult("a", PASS, "fine"), CheckResult("b", FAIL, "broken")],
        credits_spent=0.000123,
    )

    text = format_report(report)

    assert "NOT READY" in text
    assert "broken" in text
    assert "0.000123" in text


def test_run_preflight_offline_makes_no_network_calls_and_skips_live_seams(monkeypatch, pg_dsn):
    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setenv("YOUTUBE_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")

    def explode(*args, **kwargs):
        raise AssertionError("--offline must not open a network client")

    monkeypatch.setattr(httpx, "AsyncClient", explode)

    report = _run(preflight.run_preflight(offline=True))

    assert report.failed == []
    assert {r.name for r in report.skipped} >= {"Stage A · sentiment", "OpenRouter key + credit"}
    assert report.ready is False  # skips are unproven, not passing
