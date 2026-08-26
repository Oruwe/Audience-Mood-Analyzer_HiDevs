"""V2 API layer: FastAPI facade over the DuckDB warehouse.

The Streamlit console never touches DuckDB directly — all reads go through
this service, so storage can evolve independently of the UI.

Run:
    uvicorn server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schemas import EnrichedCommentRecord
from storage.db import query_enriched_records

REPO_ROOT = Path(__file__).resolve().parent
METRICS_PATH = REPO_ROOT / "data" / "eval_metrics.json"

app = FastAPI(title="Audience Mood Analyzer API", version="2.0.0")

# The Operations Console (default port 8501) is the only permitted cross-origin client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/comments", response_model=list[EnrichedCommentRecord])
def comments(limit: int = Query(default=100, ge=1, le=500)) -> list[EnrichedCommentRecord]:
    try:
        return query_enriched_records(limit=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="warehouse not initialised") from None


@app.get("/metrics")
def metrics() -> dict:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
