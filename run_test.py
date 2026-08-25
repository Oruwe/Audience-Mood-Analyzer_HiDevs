#!/usr/bin/env python3
"""Standalone Phase-1 verification harness for the ingestion layer.

Locates the project root, imports ingestion.normalizer.sanitize_comment_text
and ingestion.dedup.ContentDeduplicator, then runs async verification checks.

Usage:
    python run_test.py

Exit code 0 = all checks passed, 1 = at least one failure (CI-friendly).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path


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


async def main() -> int:
    print("=" * 60)
    print("Phase 1 verification: normalizer + deduplicator")
    print(f"project root: {REPO_ROOT}")
    print("=" * 60)

    verify_normalizer()
    await verify_deduplicator()

    passed = sum(ok for _, ok in _results)
    total = len(_results)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("All Phase-1 ingestion contracts verified ✅")
    else:
        print("Some checks FAILED — see ❌ rows above.")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
