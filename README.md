# Audience Mood Analyzer

Real-time social-listening pipeline: ingests comments from multiple platforms,
analyses sentiment / intent / urgency via an LLM failover chain, detects
emerging crises statistically, and surfaces everything on a live dashboard —
built entirely on a zero-dollar, open-source stack.

## Architecture

```
Ingestion ──▶ Broker ─────▶ Queue ──▶ Workers ──▶ DuckDB ──▶ Streamlit UI
(mock | Bluesky)  dedup +              │           ▲         (autorefresh)
Jetstream WS      PII mask             │           │
                                       ├─▶ Embedder ──▶ Theme clustering
                                       └─▶ Anomaly radar ──▶ Crisis alerts
```

| Layer | Module | Tech |
|---|---|---|
| Ingestion | `ingestion/` | asyncio generators, Jetstream WebSocket, dedup + normalisation |
| Analysis | `engine/llm_client.py` | LiteLLM: Gemini primary → Groq fallback, Pydantic-enforced JSON |
| Storage | `storage/db.py` | DuckDB, auto-migrating schema, short-lived connections |
| Radar | `engine/anomaly_detector.py`, `engine/topic_cluster.py` | numpy + KMeans + LLM theme labelling |
| Evals | `evals/benchmark.py` | scikit-learn accuracy / confusion matrix |
| Observability | `engine/llm_client.py` | Langfuse success/failure callbacks on every LLM call |
| UI | `app.py` | Streamlit + 10 s autorefresh |

## Setup

```sh
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

Required: `GEMINI_API_KEY`, `GROQ_API_KEY`.
Optional: `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (tracing), `REDIS_URL` (distributed dedup).

## Usage

```sh
python pipeline.py --mode mock                                # simulated stream
python pipeline.py --mode live --keywords ai tech marketing   # Bluesky firehose
streamlit run app.py                                          # separate terminal

python -m evals.benchmark     # sentiment benchmark -> data/eval_metrics.json
python run_test.py            # hermetic contract tests (no network, no cost)
python debug_models.py        # verify both providers respond
python recover_legacy.py      # restore archived pre-migration warehouse rows
```

## Guarantees

- PII (emails, phones) masked and tracking params stripped **before** fingerprinting or storage.
- Each distinct `(platform, text)` is analysed once per dedup TTL window.
- Token bucket caps LLM throughput (default 10/min) to respect free-tier quotas.
- Provider failover with structured-output validation; an LLM failure never kills a worker.
- Schema upgrades archive legacy tables automatically; readers trigger best-effort migration.
