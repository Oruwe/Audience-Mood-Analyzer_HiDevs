"""L2 contract test — SPEC §10 invariant 1.

"The LLM never receives a secret, token, or another user's data."
Enforced by construction: engine.llm_client's prompt builder takes only a
RawComment (SPEC: "(comments, config) — no DB handle, no session object").
Proven here two ways: (a) the function signature genuinely can't reach a
secrets store or another user's session, and (b) a real secret placed in the
environment never shows up in the messages actually sent to the model.
"""

import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace

import engine.llm_client as llm_client
from schemas import RawComment

_VALID_ANALYSIS_JSON = """
{"sentiment": "positive", "confidence": 0.9, "primary_intent": "praise_endorsement",
 "urgency_score": 0.1, "emotional_drivers": ["joy"], "summary": "Viewer liked it.",
 "recommended_action": "ignore", "suggested_reply_draft": null,
 "brand_safety_flag": false}
"""


def test_analyze_comment_signature_has_no_db_or_session_handle():
    params = inspect.signature(llm_client.analyze_comment).parameters
    forbidden_substrings = ("db", "session", "conn", "store", "secret", "user")
    for name in params:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), (
            f"analyze_comment() takes a suspicious parameter {name!r} — the "
            "prompt builder must only ever see (comment, config)"
        )


def test_environment_secret_never_reaches_the_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key-for-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "sk-should-never-leak-12345")

    captured: dict[str, object] = {}

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        captured["messages"] = messages
        captured["api_key"] = api_key
        message = SimpleNamespace(content=_VALID_ANALYSIS_JSON)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_client, "acompletion", fake_acompletion)

    comment = RawComment(
        id="c1",
        platform="youtube",
        text="Great video, thanks!",
        author_handle="@viewer",
        timestamp=datetime.now(timezone.utc),
    )
    asyncio.run(llm_client.analyze_comment(comment))

    prompt_text = " ".join(str(m["content"]) for m in captured["messages"])
    assert "sk-should-never-leak-12345" not in prompt_text
    # The provider API key is a separate transport-level kwarg, never a
    # value interpolated into the text the model reads.
    assert captured["api_key"] not in prompt_text
