"""SPEC §11 Track 1 — Stage A sentiment accuracy benchmark.

Runs every case in evals/test_dataset.json through Stage A's sentiment
classifier (engine.stage_a.classify_all_sentiments), scores with the same
lenient 3-class scheme this file has always used (strict accuracy on
positive/neutral/negative; 'mixed' ground truth counts as correct when the
model predicts either polarity), prints a classification report, and
persists metrics to data/eval_metrics.json (SPEC §6: "commit the output").

Rewritten for the §4.1 amendment (config/models.py): Stage A now classifies
via OpenRouter, batched, not the old one-comment-per-call local/Gemini
router (engine.llm_client.analyze_comment no longer exists — that name is
now Stage B's batched classify_batch/classify_all, a different job
entirely). Needs OPENROUTER_API_KEY. Batching also means the old
MAX_CONCURRENCY/STAGGER_SECONDS free-tier throttling is gone: this dataset
is 1-2 sentiment batches now, not 30 individual rate-limited calls.

Usage (from repo root):
    python -m evals.benchmark

Also importable and callable directly with `persist_to_file=False` — app.py's
"Model accuracy benchmark" panel does exactly this, running the same real
benchmark against the live deployed model on demand and persisting the
result to Postgres (storage.postgres.save_eval_run) instead of a file, so
it survives a redeploy/restart the same way SPEC §8 job state already does.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make repo-root imports work regardless of how the script is launched.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from engine.stage_a import classify_all_sentiments  # noqa: E402
from schemas import RawComment  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.json"
METRICS_PATH = REPO_ROOT / "data" / "eval_metrics.json"

COARSE_LABELS = ["positive", "neutral", "negative"]            # 3-class eval space
VALID_EXPECTED = {"positive", "neutral", "negative", "mixed"}  # dataset label set


def load_dataset() -> list[dict]:
    """Load the dataset and fail fast if any expected label drifts from the label set."""
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    bad = [
        (i, c.get("expected_mood"))
        for i, c in enumerate(cases)
        if c.get("expected_mood") not in VALID_EXPECTED
    ]
    if bad:
        raise SystemExit(f"Invalid expected_mood values in {DATASET_PATH}: {bad}")
    return cases


def _coarse(sentiment_value: str) -> str:
    """Map Stage A's 5-class sentiment onto the 3-class eval space."""
    return {"strongly_positive": "positive", "critical_escalation": "negative"}.get(
        sentiment_value, sentiment_value
    )


def _comment_id(index: int) -> str:
    return f"eval-{index:03d}"


async def run_benchmark(*, persist_to_file: bool = True) -> dict:
    cases = load_dataset()
    print("=== SPEC §11 Track 1: Stage A sentiment benchmark ===")
    print(f"Loaded {len(cases)} cases from {DATASET_PATH}")

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is required — Stage A classifies via OpenRouter "
            "now (SPEC.md §4.1 amendment), not a local encoder or Gemini."
        )

    comments = [
        RawComment(
            id=_comment_id(i),
            platform=case["platform"],
            text=case["text"],
            author_id="eval_harness",
            timestamp=datetime.now(timezone.utc),
        )
        for i, case in enumerate(cases)
    ]

    error: str | None = None
    try:
        results_by_id = await classify_all_sentiments(comments, api_key=api_key)
    except Exception as exc:  # noqa: BLE001 — record and report, don't crash the run
        results_by_id = {}
        error = f"{type(exc).__name__}: {exc}"

    metrics: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "failed_cases": len(cases) - len(results_by_id),
        "labels": COARSE_LABELS,
    }

    if not results_by_id:
        print(f"\nBenchmark failed entirely — {error}\nNo metrics written.")
        metrics.update({"accuracy": None, "accuracy_lenient": None,
                        "mixed_cases": 0, "mixed_covered": 0,
                        "confusion_matrix": None, "classification_report": None,
                        "error": error})
    else:
        scored: list[tuple[str, str | None]] = []
        for i, case in enumerate(cases):
            item = results_by_id.get(_comment_id(i))
            predicted = _coarse(item.sentiment.value) if item is not None else None
            status = "OK  " if item is not None else "FAIL"
            print(f"[{i + 1:>2}/{len(cases)}] {status}  "
                  f"expected={case['expected_mood']:<8} predicted={predicted or '-'}")
            scored.append((case["expected_mood"], predicted))

        strict_pairs = [(e, p) for e, p in scored if e != "mixed" and p is not None]
        mixed_preds = [p for e, p in scored if e == "mixed" and p is not None]
        scored_count = sum(1 for _, p in scored if p is not None)

        y_true = [e for e, _ in strict_pairs]
        y_pred = [p for _, p in strict_pairs]

        accuracy_strict: float | None = None
        cm = None
        report_dict = None
        if y_true:
            accuracy_strict = float(accuracy_score(y_true, y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=COARSE_LABELS)
            report_dict = classification_report(
                y_true, y_pred, labels=COARSE_LABELS, output_dict=True, zero_division=0
            )
            print("\n--- Classification Report (3-class, 'mixed' excluded) ---")
            print(classification_report(
                y_true, y_pred, labels=COARSE_LABELS, digits=3, zero_division=0))
            print(f"Confusion matrix (rows=true, cols=predicted): {COARSE_LABELS}")
            for label, row in zip(COARSE_LABELS, cm.tolist()):
                print(f"  {label:<9}{row}")

        mixed_correct = sum(1 for p in mixed_preds if p in {"positive", "negative"})
        strict_correct = sum(1 for e, p in strict_pairs if e == p)
        accuracy_lenient = (
            (strict_correct + mixed_correct) / scored_count if scored_count else None
        )

        if accuracy_strict is not None:
            print(f"\nStrict accuracy (non-mixed): {accuracy_strict:.3f} "
                  f"({len(strict_pairs)} scored)")
        else:
            print("\nStrict accuracy: n/a — no case scored")
        if accuracy_lenient is not None:
            print(f"Lenient accuracy (mixed counts if pos/neg): {accuracy_lenient:.3f}")
        print(f"Mixed cases: {mixed_correct}/{len(mixed_preds)} resolved to a polarity")

        metrics.update({
            "accuracy": accuracy_strict,          # dashboard keeps reading this key
            "accuracy_lenient": accuracy_lenient,
            "mixed_cases": len(mixed_preds),
            "mixed_covered": mixed_correct,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "classification_report": report_dict,
        })
        if error:
            metrics["error"] = error  # partial-failure case: some results, some not

    if persist_to_file:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics saved -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    asyncio.run(run_benchmark())
