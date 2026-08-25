#!/usr/bin/env python3
"""Standalone verification harness: Phase-1 ingestion contracts plus a fully
hermetic Phase-4 radar suite (temp DuckDB file, stubbed litellm — no network,
no real LLM calls, no cost).

Locates the project root, imports ingestion.normalizer.sanitize_comment_text,
ingestion.dedup.ContentDeduplicator, engine.anomaly_detector.detect_anomalies
and engine.topic_cluster.extract_trending_theme, then runs async checks.

Usage:
    python run_test.py

Exit code 0 = all checks passed, 1 = at least one failure (CI-friendly).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def _find_repo_root() -> Path:
    """Walk up from this file until a directory containing ingestion/ is found."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "ingestion" / "normalizer.py").exists():
            return candidate
    return here


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ingestion.dedup import ContentDeduplicator, fingerprint_text
    from ingestion.normalizer import sanitize_comment_text
except ImportError as exc:
    print(f"❌ Could not import ingestion modules from {REPO_ROOT}: {exc}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

PASS, FAIL = "✅ PASS", "❌ FAIL"
_results: list[tuple[str, bool]] = []


def check(name: str, condition: object, detail: str = "") -> None:
    ok = bool(condition)
    _results.append((name, ok))
    suffix = f"   [{detail}]" if detail else ""
    print(f"  {PASS if ok else FAIL}  {name}{suffix}")


def verify_normalizer() -> None:
    print("\n─── sanitize_comment_text ──────────────────────────────────")

    out = sanitize_comment_text(
        "Full write-up: https://example.com/blog/post?utm_source=x&utm_medium=cpc&id=42 🎉"
    )
    check("URL: UTM stripped, real params kept",
          "https://example.com/blog/post?id=42" in out and "utm_" not in out, out)

    out = sanitize_comment_text("Questions? Email jane.doe+news@sub.example.co.uk")
    check("PII: email masked",
          "[EMAIL]" in out and "@" not in out.replace("[EMAIL]", ""), out)

    out = sanitize_comment_text("Ring support on +1 555 123 4567 or 555-867-5309.")
    check("PII: both phone formats masked",
          out.count("[PHONE]") == 2 and "555" not in out, out)

    out = sanitize_comment_text("Too     many    spaces\tand\ttabs\n\n\n\n\nthen more")
    check("whitespace collapsed, blank runs capped",
          "  " not in out and "\t" not in out and "\n\n\n" not in out, repr(out))

    out = sanitize_comment_text('   "What a great launch!"   ')
    check("leading/trailing quotes stripped", out == "What a great launch!", repr(out))


async def verify_deduplicator() -> None:
    print("\n─── ContentDeduplicator (in-memory LRU fallback) ───────────")

    saved = os.environ.pop("REDIS_URL", None)  # force deterministic memory backend
    try:
        dedup = ContentDeduplicator(max_size=500, ttl_seconds=3600)

        fp_a = fingerprint_text("  This Product Is Amazing  ")
        fp_b = fingerprint_text("this product is amazing")
        check("fingerprint is 64-char sha256 hex",
              len(fp_a) == 64 and all(ch in "0123456789abcdef" for ch in fp_a), fp_a)
        check("case/whitespace-insensitive fingerprinting", fp_a == fp_b)

        check("first sighting  -> not duplicate", await dedup.is_duplicate(fp_a) is False)
        check("second sighting -> duplicate",     await dedup.is_duplicate(fp_a) is True)
        check("text variant resolves to same fingerprint", await dedup.is_duplicate(fp_b) is True)

        distinct = [fingerprint_text(f"genuinely different comment #{i}") for i in range(3)]
        check("distinct texts -> distinct fingerprints", len(set(distinct)) == 3)
        for fp in distinct:
            await dedup.is_duplicate(fp)
        repeat_hits = [await dedup.is_duplicate(fp) for fp in distinct]
        check("independent texts each flag on repeat", all(repeat_hits))
        await dedup.aclose()

        # --- LRU eviction (max_size=3) ---
        small = ContentDeduplicator(max_size=3, ttl_seconds=3600)
        fps = [fingerprint_text(f"evict-me-{i}") for i in range(4)]
        for fp in fps[:3]:
            await small.is_duplicate(fp)
        await small.is_duplicate(fps[3])  # 4th insert evicts fps[0]
        check("LRU evicts oldest beyond max_size",
              await small.is_duplicate(fps[0]) is False)
        check("recent entries still remembered",
              await small.is_duplicate(fps[3]) is True)
        await small.aclose()

        # --- TTL expiry ---
        flash = ContentDeduplicator(max_size=10, ttl_seconds=0)  # instantly stale
        fp = fingerprint_text("ttl probe")
        await flash.is_duplicate(fp)
        check("expired entries treated as fresh", await flash.is_duplicate(fp) is False)
        await flash.aclose()
    finally:
        if saved is not None:
            os.environ["REDIS_URL"] = saved


async def verify_phase4_radar() -> None:
    print("\n─── Phase 4 radar: anomaly detector + theme cluster ────────")

    try:
        import duckdb
        import litellm as litellm_mod
        import storage.db as db_mod
        from engine.anomaly_detector import detect_anomalies
        from engine.topic_cluster import extract_trending_theme
        from schemas import (
            EnrichedCommentRecord,
            PrimaryIntent,
            RecommendedAction,
            Sentiment,
        )
    except ImportError as exc:
        check("phase-4 imports available", False, str(exc))
        return

    tmp = tempfile.TemporaryDirectory()
    original_db_path = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name) / "radar_hermetic.duckdb"

    def synth(i: int, urgency: float, summary: str) -> EnrichedCommentRecord:
        return EnrichedCommentRecord(
            comment_id=f"radar-{i:03d}",
            platform="bluesky",
            author_handle=f"@radar_tester_{i}",
            raw_text=f"synthetic radar comment {i}",
            sentiment=Sentiment.NEGATIVE,
            confidence=0.92,
            primary_intent=PrimaryIntent.BUG_REPORT,
            urgency_score=urgency,
            emotional_drivers=["frustration"],
            summary=summary,
            recommended_action=RecommendedAction.ESCALATE_TO_SUPPORT,
            suggested_reply_draft=None,
            brand_safety_flag=False,
            embedding=None,
            cluster_id=None,
            latency_ms=11.0,
            model_used="hermetic",
            processed_at=datetime.now(timezone.utc),
        )

    def clear_window() -> None:
        with duckdb.connect(str(db_mod.DB_PATH)) as conn:
            conn.execute("DELETE FROM analyzed_comments")

    llm_calls = {"count": 0}

    async def fake_acompletion(*args, **kwargs):
        llm_calls["count"] += 1
        message = SimpleNamespace(content='  "Login Outage Storm"  ')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    real_acompletion = litellm_mod.acompletion
    litellm_mod.acompletion = fake_acompletion
    saved_gemini_key = os.environ.pop("GEMINI_API_KEY", None)

    try:
        # -- empty warehouse -------------------------------------------------
        check("empty warehouse -> no alert", await detect_anomalies() is None)

        # -- calm window -----------------------------------------------------
        for i in range(4):
            await db_mod.insert_enriched_record(synth(i, 0.20, f"calm note {i}"))
        check("calm window -> no alert", await detect_anomalies() is None)
        clear_window()  # isolate next window: mean-trigger must see only hot rows

        # -- mean-trigger CRITICAL + stubbed theme summarisation --------------
        for i in range(4, 8):
            await db_mod.insert_enriched_record(synth(i, 0.95, f"login broken {i}"))
        alert = await detect_anomalies()
        check("hot window raises alert", alert is not None)
        if alert is not None:
            check("mean-trigger severity is CRITICAL",
                  alert.severity == "CRITICAL", alert.severity)
            check("alert covers whole window", len(alert.affected_comment_ids) == 4)
            check("theme via stubbed LLM, quotes stripped",
                  alert.theme == "Login Outage Storm", alert.theme)
            check("trigger reason carries stats", "mean urgency" in alert.trigger_reason)
        clear_window()

        # -- WARNING path: mean <= 0.70 but >= 3 hard spikes -------------------
        # NB: 0.05 (not 0.10) keeps the float mean at 0.6875, safely below the
        # strict "> 0.70" mean trigger — 0.10 floats to 0.7000000000000001.
        for i, u in enumerate([0.90, 0.90, 0.90, 0.05]):
            await db_mod.insert_enriched_record(synth(100 + i, u, f"spike note {i}"))
        alert = await detect_anomalies()
        check("spike-only window still alerts", alert is not None)
        if alert is not None:
            check("spike-trigger severity is WARNING",
                  alert.severity == "WARNING", alert.severity)
        clear_window()

        # -- theme cluster: below minimum cluster size -> fallback, no LLM -----
        pair = [synth(200, 0.99, "a"), synth(201, 0.98, "b")]
        calls_before = llm_calls["count"]
        check("tiny batch -> fallback theme",
              await extract_trending_theme(pair) == "General Feedback")
        check("tiny batch made zero LLM calls", llm_calls["count"] == calls_before)

        # -- theme cluster: API-key pre-guard -----------------------------------
        trio = [synth(300, 0.97, "s1"), synth(301, 0.96, "s2"), synth(302, 0.95, "s3")]
        calls_before = llm_calls["count"]
        check("missing GEMINI_API_KEY -> fallback, no call",
              await extract_trending_theme(trio) == "General Feedback"
              and llm_calls["count"] == calls_before)

        # -- theme cluster: happy path through the stub --------------------------
        os.environ["GEMINI_API_KEY"] = "hermetic-fake-key"
        theme = await extract_trending_theme(trio)
        check("key present -> stubbed theme returned",
              theme == "Login Outage Storm", theme)
        check("happy path hit the LLM exactly once",
              llm_calls["count"] == calls_before + 1)
    finally:
        litellm_mod.acompletion = real_acompletion
        if saved_gemini_key is not None:
            os.environ["GEMINI_API_KEY"] = saved_gemini_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)
        db_mod.DB_PATH = original_db_path
        tmp.cleanup()


async def main() -> int:
    print("=" * 60)
    print("Verification: Phase-1 ingestion + Phase-4 radar (hermetic)")
    print(f"project root: {REPO_ROOT}")
    print("=" * 60)

    verify_normalizer()
    await verify_deduplicator()
    await verify_phase4_radar()

    passed = sum(ok for _, ok in _results)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("All Phase-1 + Phase-4 contracts verified ✅")
    else:
        print("Some checks FAILED — see ❌ rows above.")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
