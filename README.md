# Audience Mood Analyzer — V2

[![CI](https://github.com/Oruwe/Audience-Mood-Analyzer_HiDevs/actions/workflows/ci.yml/badge.svg)](https://github.com/Oruwe/Audience-Mood-Analyzer_HiDevs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time social-listening platform with a decoupled three-process
architecture: an asyncio pipeline ingests and analyses the firehose, a FastAPI
service owns every read of the DuckDB warehouse, and a Streamlit Operations
Console consumes the REST API. Zero-dollar stack throughout.

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│   Pipeline   │───▶│ DuckDB warehouse │◀───│  FastAPI backend     │
│ ingest +     │    │ data/analytics.  │    │  server.py  :8000    │
│ analyse      │    │ duckdb           │    │  /comments  /metrics │
└──────┬───────┘    └──────────────────┘    └──────────┬───────────┘
       │                                               │ HTTP/JSON
       │ tiered LLM routing                            ▼
       ▼                                      ┌──────────────────────┐
 Gemini flash ─▶ Groq gpt-oss-20b             │ Operations Console   │
 (Pydantic-enforced JSON)                     │ app.py  :8501        │
                                              └──────────────────────┘
```

| Layer | Module | Tech |
|---|---|---|
| Ingestion | `ingestion/` | asyncio generators, Jetstream WebSocket, dedup + PII masking |
| Analysis | `engine/llm_client.py` | LiteLLM tiered routing, structured outputs |
| Persistence | `storage/db.py` | DuckDB, auto-migrating schema, short-lived connections |
| API | `server.py` | FastAPI + CORS — the console's only data surface |
| Radar | `engine/anomaly_detector.py`, `engine/topic_cluster.py` | numpy + KMeans + LLM theme labelling |
| Evals | `evals/benchmark.py` | scikit-learn accuracy / confusion matrix |
| Observability | `engine/llm_client.py` | Langfuse success/failure callbacks on every LLM call |
| Console | `app.py` | Streamlit + 10 s autorefresh over the REST API |

### Multi-tier LLM routing

1. **Analysis:** `gemini/gemini-3.6-flash` primary → `groq/openai/gpt-oss-20b` fallback; every response validated against a Pydantic contract, and a provider failure never kills a worker.
2. **Embeddings:** `gemini/text-embedding-004` → deterministic local hash-vector fallback (offline-safe).
3. **Theme labelling:** Gemini with a statistical fallback label when the key is absent or the call fails.

## Setup

```sh
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

Required: `GEMINI_API_KEY`, `GROQ_API_KEY`.
Optional: `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (tracing), `REDIS_URL` (distributed dedup), `API_BASE_URL` (console → API, defaults to `http://127.0.0.1:8000`).

## Running the stack concurrently

Three processes, three terminals:

```sh
uvicorn server:app --host 127.0.0.1 --port 8000   # T1 — API backend
python pipeline.py --mode mock                    # T2 — ingestion + analysis
streamlit run app.py                              # T3 — Operations Console
```

Single shell (bash/Linux/macOS):

```sh
uvicorn server:app --port 8000 & python pipeline.py --mode mock & streamlit run app.py
```

On Windows, run the three commands in separate terminals (or prefix each with `start`).

Live mode: `python pipeline.py --mode live --keywords ai tech marketing`

## Tooling

```sh
python -m evals.benchmark                       # sentiment benchmark -> data/eval_metrics.json
pytest                                          # hermetic unit tests (no network, no cost)
python scripts/check_providers.py               # verify both providers respond
python scripts/maintenance/recover_legacy.py    # restore archived pre-migration warehouse rows
python scripts/maintenance/repair_warehouse.py  # re-serialize rows through the hardened reader
curl http://127.0.0.1:8000/health               # API smoke test
```

## Development

```sh
pip install -r requirements-dev.txt   # adds pytest, pytest-asyncio, ruff on top of the app deps
ruff check .                          # lint
pytest -q                             # 67 hermetic tests: normalizer, dedup, engine, storage, API
```

CI (`.github/workflows/ci.yml`) runs both on every push/PR to `main`. Neither
touches a real Gemini/Groq/Langfuse endpoint or a shared DuckDB file — every
test that needs a provider response stubs it, and every test that needs a
warehouse gets its own `tmp_path` DuckDB file, so the suite has no network
dependency and no API cost. `python -m evals.benchmark` is intentionally not
part of CI: it calls the real providers to score live model accuracy, which
needs API keys and isn't hermetic by design.

## Guarantees

- PII (emails, phones) masked and tracking params stripped **before** fingerprinting or storage.
- Each distinct `(platform, text)` is analysed once per dedup TTL window.
- Token bucket caps LLM throughput (default 10/min) to respect free-tier quotas.
- Tiered provider failover with structured-output validation.
- Schema upgrades archive legacy tables automatically; readers trigger best-effort migration.
- The console holds **no** database credentials — all access flows through the API.

## License

[MIT](LICENSE)
