"""Streamlit dashboard: live view of analysed comments + benchmark accuracy.

Run alongside the pipeline:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from storage.db import query_analyzed_comments

METRICS_PATH = Path("data/eval_metrics.json")
REFRESH_MS = 10_000  # poll the warehouse every 10 s while ingestion runs


@st.cache_data(ttl=30, show_spinner=False)
def load_eval_metrics() -> dict[str, Any]:
    """Read the latest benchmark output; empty dict when the file is absent."""
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _extract_accuracy(metrics: dict[str, Any]) -> float | None:
    val = metrics.get("accuracy")
    return float(val) if isinstance(val, (int, float)) else None


@st.cache_data(ttl=5, show_spinner="Querying DuckDB…")
def load_analyzed_comments(limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        records = query_analyzed_comments(limit=limit)
    except FileNotFoundError:
        return [], "No warehouse yet — run `python pipeline.py --mode mock` first."
    except Exception as exc:
        return [], f"Cannot open warehouse: {type(exc).__name__}: {exc}"
    return [r.model_dump(mode="json") for r in records], None


def build_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records([
        {
            "Time": r["processed_at"],
            "Platform": r.get("platform", "—"),
            "Author": r.get("author", "—"),
            "Comment": r.get("comment_text", ""),
            "Summary": r["summary"],
            "Mood": r["mood"],
            "Confidence": r["confidence"],
            "Urgency": r["urgency_score"],
            "Action": r["marketing_action"],
        }
        for r in rows
    ])
    if not df.empty:
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
        df = df.sort_values("Time", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Social Listening Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Social Listening Dashboard")
    st.caption(f"Auto-refreshes every {REFRESH_MS // 1000} s while ingestion runs.")

    st_autorefresh(interval=REFRESH_MS, key="dashboard_autorefresh")

    limit = st.sidebar.slider("Rows to display", min_value=25, max_value=500,
                              value=100, step=25)

    accuracy = _extract_accuracy(load_eval_metrics())
    rows, db_error = load_analyzed_comments(limit)
    if db_error:
        st.error(db_error)
    df = build_frame(rows)
    avg_urgency = float(df["Urgency"].mean()) if not df.empty else None

    c1, c2 = st.columns(2)
    c1.metric(
        "Benchmark accuracy",
        f"{accuracy:.1%}" if accuracy is not None else "n/a",
        help="Run `python -m evals.benchmark` to generate data/eval_metrics.json.",
    )
    c2.metric(
        "Average urgency (shown rows)",
        f"{avg_urgency:.2f}" if avg_urgency is not None else "n/a",
    )

    if avg_urgency is not None:
        st.progress(min(max(avg_urgency, 0.0), 1.0),
                    text=f"Average urgency: {avg_urgency:.2f}")

    st.subheader(f"Analysed comments ({len(df)})")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comment": st.column_config.TextColumn(width="large"),
            "Summary": st.column_config.TextColumn(width="medium"),
            "Confidence": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"),
            "Urgency": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )


if __name__ == "__main__":
    main()
