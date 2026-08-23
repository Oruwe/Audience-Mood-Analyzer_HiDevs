"""Async LLM routing layer: Gemini primary -> Groq fallback, traced via Langfuse."""

import os
import litellm
from litellm import Router
from schemas import AnalyzedMood, MoodAnalysis, RawComment

# --- Observability: every routed call emits a Langfuse trace ---
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

SYSTEM_PROMPT = (
    "You are a social-media listening analyst. Analyse the given comment and "
    "respond ONLY with JSON matching: "
    '{"mood": "positive|neutral|negative|mixed", "confidence": <0.0-1.0>, '
    '"summary": "<one sentence>", "urgency_score": <0.0-1.0>, '
    '"marketing_action": "none|reply|amplify|escalate"}. '
    "urgency_score reflects how quickly the brand must react."
)

_router = Router(
    model_list=[
        {
            "model_name": "mood-analyzer",
            "litellm_params": {
                "model": "gemini/gemini-1.5-flash",
                "api_key": os.environ.get("GEMINI_API_KEY", ""),
            },
        },
        {
            "model_name": "mood-analyzer-fallback",
            "litellm_params": {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.environ.get("GROQ_API_KEY", ""),
            },
        },
    ],
    fallbacks=[{"mood-analyzer": ["mood-analyzer-fallback"]}],
    num_retries=2,
)

async def analyze_comment(comment: RawComment) -> AnalyzedMood:
    """Analyse one comment; auto-falls back to Groq if Gemini fails."""
    response = await _router.acompletion(
        model="mood-analyzer",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[{comment.platform}] @{comment.author}: {comment.text}"},
        ],
        response_format=MoodAnalysis, 
        metadata={"comment_id": comment.id, "platform": comment.platform}, 
    )
    analysis = MoodAnalysis.model_validate_json(response.choices[0].message.content)
    return AnalyzedMood(comment_id=comment.id, **analysis.model_dump())