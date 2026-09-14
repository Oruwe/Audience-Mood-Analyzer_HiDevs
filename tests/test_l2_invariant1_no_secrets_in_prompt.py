"""L2 contract test — SPEC §10 invariant 1.

"The LLM never receives a secret, token, or another user's data."
Enforced by construction: engine.llm_client's Stage B prompt builder
(inside engine.batching, shared with Stage A) takes only comments + model
config (SPEC: "(comments, config) — no DB handle, no session object").
Proven here two ways: (a) the function signature genuinely can't reach a
secrets store or another user's session, and (b) a real secret placed in
the environment never shows up in the messages actually sent to the model.
"""

import asyncio
import inspect
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import engine.batching as batching
import engine.llm_client as llm_client
from schemas import RawComment

_VALID_BATCH_JSON = json.dumps({
    "results": [{"comment_id": "0", "intent": "praise", "is_request": False, "is_confusion": False}]
})


def test_classify_batch_signature_has_no_db_or_session_handle():
    params = inspect.signature(llm_client.classify_batch).parameters
    forbidden_substrings = ("db", "session", "conn", "store", "secret", "user")
    for name in params:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), (
            f"classify_batch() takes a suspicious parameter {name!r} — the "
            "prompt builder must only ever see (comments, config)"
        )


def test_environment_secret_never_reaches_the_model(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "sk-should-never-leak-12345")

    captured: dict[str, object] = {}

    async def fake_acompletion(*, model, api_key, messages, response_format, timeout):
        captured["messages"] = messages
        captured["api_key"] = api_key
        message = SimpleNamespace(content=_VALID_BATCH_JSON)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(batching, "acompletion", fake_acompletion)

    comment = RawComment(
        id="c1",
        platform="youtube",
        text="Great video, thanks!",
        author_handle="@viewer",
        timestamp=datetime.now(timezone.utc),
        video_id="v1",
    )
    asyncio.run(llm_client.classify_batch([comment], api_key="fake-openrouter-key"))

    prompt_text = " ".join(str(m["content"]) for m in captured["messages"])
    assert "sk-should-never-leak-12345" not in prompt_text
    # The provider API key is a separate transport-level kwarg, never a
    # value interpolated into the text the model reads.
    assert captured["api_key"] not in prompt_text
