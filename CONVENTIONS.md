# CONVENTIONS.md
## Directives
- **Zero-Dollar Stack:** ONLY use open-source, local, or free-tier tools. NO AWS, GCP, OpenAI paid tiers, or standard Postgres/Redis.
- **Async First:** Data ingestion and routing must use Python `asyncio`.
- **Type Safety:** Use `Pydantic` for all data schemas.

## Tech Stack
1. **Ingestion:** `asyncio` generators (Mock stream first).
2. **Execution/Routing:** `LiteLLM` for routing. Primary provider is `gemini`, fallback is `groq`. Enforce structured outputs via `instructor` or Pydantic.
3. **Storage:** `DuckDB` for state and real-time SQL analytics.
4. **Observability:** `Langfuse`. All LLM calls must include callbacks.
5. **Evals:** `scikit-learn` (accuracy, confusion matrix).
6. **UI:** `Streamlit`.

---

## Deviations from the directives above

The directives are left exactly as written; this section records where the
shipped system departs from them, and why. Each departure was measured
rather than assumed, and each is reversible — every model string lives in
`config/models.py` and nothing else names one (enforced by a test).

**Storage: Postgres, not DuckDB.** DuckDB is an in-process file database, so
its state lives on the container's local disk. Render's filesystem is
ephemeral — every redeploy and every free-tier sleep wipes it. The whole
point of checkpointing is that a crash or a sleep does not re-spend YouTube
quota and OpenRouter credit on work already done, and a warehouse that
vanishes on restart cannot deliver that. Postgres on a free Neon/Supabase
tier keeps the zero-dollar constraint intact while surviving the restart.

**Routing: OpenRouter, not Gemini→Groq.** One provider account reaches every
vendor, which is what makes the fallback chain meaningful: each stage's
backup sits on a *different* vendor from its primary and from every other
stage's backup, so no single capacity event can take out two stages' insurance
at once. Under the original two-provider scheme, a Gemini outage and a Groq
outage were the only two states the system could distinguish.

**Not free-tier inference.** This is the real departure and it was not a
preference. Every stage started on OpenRouter `:free` endpoints. A live run
spent 6+ minutes in Stage A alone, returning
`limit_source: upstream_provider_shared_pool` — free capacity is shared across
*all* OpenRouter users, so the throttling had nothing to do with our volume.
Moving Stage A to paid simply relocated the stall into Stage B. All four
stages are now paid, at roughly **$0.03 per 500-comment analysis**. The
directive's intent — do not spend real money on infrastructure — is otherwise
honoured throughout: free YouTube API tier, free Postgres tier, free hosting.

**Structured outputs via Pydantic, not `instructor`.** The directive allows
either. `litellm`'s `response_format` takes the Pydantic model directly, so
`instructor` would have been a wrapper around a call that already does the
job. It has been removed from `requirements.txt` rather than left installed
and unused.

**Langfuse is not wired.** The directive asks for callbacks on every LLM call.
There are none, so the dependency has been removed instead of shipping a
manifest that advertises tracing this project does not do. Observability is
currently structured logging plus `harness/preflight.py`. This is a genuine
gap, not a considered substitution.

**Stage A is an API call, not a local encoder.** The design spec called for a
local encoder — free, deterministic, and structurally immune to prompt
injection. It does not fit the deployment: Render's free plan gives 512 MB RAM
and 0.1 CPU, the app already uses ~440 MB, and PyTorch alone is 300–500 MB
resident before any weights load. See `config/models.py` for the full
accounting and the conditions under which this should be revisited.
