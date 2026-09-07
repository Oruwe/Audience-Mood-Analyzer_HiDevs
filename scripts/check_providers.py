#!/usr/bin/env python3
"""Diagnostic: verify every provider in the live model chain is reachable.

Replaces the two ad-hoc scripts (``debug.py``, ``debug_models.py``) that
accumulated at the repo root during development — one checked only Groq,
the other hardcoded its own copy of the model list. That second copy is
exactly the kind of duplication that goes stale: ``engine/llm_client.py``
itself carries a comment warning that a legacy Llama endpoint went
enterprise-restricted and must not be referenced. This script reads
``_MODEL_CHAIN`` directly from ``engine.llm_client`` instead of hardcoding
a second list, so it can never drift from what the app actually calls.

Usage (from repo root):
    python scripts/check_providers.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
litellm.suppress_debug_info = True

from engine.llm_client import _MODEL_CHAIN  # noqa: E402


async def check_one(model: str, key_env: str) -> bool:
    api_key = os.environ.get(key_env, "")
    if not api_key:
        print(f"❌ {model}: {key_env} is empty "
              "(check for a stray .env.txt if you're on Windows)")
        return False
    try:
        await litellm.acompletion(
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
            timeout=15,
        )
        print(f"✅ {model}: {key_env} loaded ({api_key[:4]}...) and the call succeeded")
        return True
    except Exception as exc:  # noqa: BLE001 — report every provider, don't stop at the first
        print(f"❌ {model}: call failed — {type(exc).__name__}: {exc}")
        return False


async def main() -> int:
    print("=== Provider check (engine.llm_client._MODEL_CHAIN) ===\n")
    results = [await check_one(model, key_env) for model, key_env in _MODEL_CHAIN]
    ok = sum(results)
    print(f"\n{ok}/{len(results)} provider(s) reachable.")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
