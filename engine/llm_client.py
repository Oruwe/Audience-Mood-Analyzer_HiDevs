"""Async LLM client: Gemini primary, Groq fallback, Pydantic-enforced output."""

import logging
import os
import time

import litellm
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()  # MUST run before reading API keys

from schemas import DeepMoodAnalysis, EnrichedCommentRecord, RawComment  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert social-media listening analyst. Analyse the comment and "
    "respond ONLY with JSON matching exactly this schema:\n"
    '{"sentiment": "strongly_positive|positive|neutral|negative|critical_escalation", '
    '"confidence": <0.0-1.0>, '
    '"primary_intent": "bug_report|feature_request|pricing_complaint|'
    'praise_endorsement|churn_risk|general_inquiry|sarcastic_troll", '
    '"urgency_score": <0.0-1.0>, '
    '"emotional_drivers": ["<short phrase>", ...], '
    '"summary": "<one sentence, max 200 chars>", '
    '"recommended_action": "ignore|community_reply|escalate_to_support|'
    'escalate_to_pr|amplify_marketing", '
    '"suggested_reply_draft": "<draft reply or null>", '
    '"brand_safety_flag": <true|false>}\n'
    "Guidance: urgency_score reflects how fast the brand must react "
    "(critical_escalation/churn_risk => high). Set brand_safety_flag=true only for "
    "legal, safety, or harassment exposure. suggested_reply_draft must be null "
    "unless recommended_action implies a reply."
)

# Ordered failover chain per CONVENTIONS.md (gemini primary, groq fallback).
# gpt-oss-20b is the active Groq developer-tier model; the legacy Llama
# endpoints are enterprise-restricted and must not be referenced here.
_MODEL_CHAIN: list[tuple[str, str]] = [
    ("gemini/gemini-3.6-flash", "GEMINI_API_KEY"),
    ("groq/openai/gpt-oss-20b", "GROQ_API_KEY"),
]

# ---------------------------------------------------------------------------
# Observability (CONVENTIONS.md): route every LLM call through Langfuse.
# Callbacks activate only when credentials exist, so a missing key degrades to
# a startup warning instead of failing every completion.
# ---------------------------------------------------------------------------
_LANGFUSE_ENABLED = bool(
    os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
)
if _LANGFUSE_ENABLED:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    logger.info("Langfuse tracing enabled (success + failure callbacks)")
else:
    logger.warning("Langfuse keys missing — LLM tracing disabled")


async def analyze_comment(comment: RawComment) -> EnrichedCommentRecord:
    """Analyse one comment, trying each provider in order until one succeeds."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"[{comment.platform}] @{comment.author_handle or comment.author_id}: {comment.text}"},
    ]
    last_exc: Exception | None = None
    for model, key_env in _MODEL_CHAIN:
        api_key = os.environ.get(key_env, "")
        if not api_key:
            continue
        started = time.perf_counter()
        try:
            response = await acompletion(
                model=model,
                api_key=api_key,
                messages=messages,
                response_format=DeepMoodAnalysis,
                timeout=30,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            analysis = DeepMoodAnalysis.model_validate_json(
                response.choices[0].message.content
            )
            return EnrichedCommentRecord(
                **analysis.model_dump(),
                comment_id=comment.id,
                platform=comment.platform,
                author_handle=comment.author_handle,
                raw_text=comment.text,
                latency_ms=latency_ms,
                model_used=model,
            )
        except Exception as exc:  # noqa: BLE001 — failover, never crash the worker
            last_exc = exc
            logger.warning("LLM call failed on %s (%s); trying next provider", model, exc)
    raise RuntimeError(f"All LLM providers failed; last error: {last_exc}") from last_exc


def flush_observability() -> None:
    """Best-effort Langfuse flush at shutdown (no-op when tracing disabled)."""
    if not _LANGFUSE_ENABLED:
        return
    try:
        from langfuse import Langfuse  # lazy: keeps the hot path import-free

        Langfuse().flush()
    except Exception as exc:  # noqa: BLE001 — telemetry must never break shutdown
        logger.debug("Langfuse flush skipped (%s)", exc)


# Alias kept for Phase 3 API compatibility.
analyze_comment_deep = analyze_comment
