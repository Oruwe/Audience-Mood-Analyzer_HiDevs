"""Async LLM routing layer: Groq Primary."""

import os
import litellm
from litellm import Router
from dotenv import load_dotenv

# Force environment variables to load
load_dotenv()

from schemas import AnalyzedMood, MoodAnalysis, RawComment

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
                "model": "groq/llama-3.3-70b-versatile", # Groq's active flagship model
                "api_key": os.environ.get("GROQ_API_KEY", ""),
            },
        }
    ],
    num_retries=2,
)

async def analyze_comment(comment: RawComment) -> AnalyzedMood:
    """Analyse one comment using Groq."""
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