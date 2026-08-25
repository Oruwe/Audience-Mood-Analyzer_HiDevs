"""Async LLM client: Direct Gemini execution (Simplified)."""

import os
from litellm import acompletion
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

async def analyze_comment(comment: RawComment) -> AnalyzedMood:
    """Analyse one comment using Gemini directly."""
    response = await acompletion(
        model="gemini/gemini-3.6-flash", # Google's active 2026 model!
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[{comment.platform}] @{comment.author}: {comment.text}"},
        ],
        response_format=MoodAnalysis, 
    )
    
    # Parse the strict JSON response into our Pydantic schema
    analysis = MoodAnalysis.model_validate_json(response.choices[0].message.content)
    return AnalyzedMood(comment_id=comment.id, **analysis.model_dump())