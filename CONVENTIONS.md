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