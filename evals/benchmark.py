"""Node 5 — Evaluation layer.

Runs every case in evals/test_dataset.json through the production router
(engine.llm_client.analyze_comment), scores mood detection with scikit-learn,
prints a classification report, and persists raw metrics to
data/eval_metrics.json for the Streamlit dashboard.

Usage (from repo root):
    python -m evals.benchmark
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
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

from engine.llm_client import analyze_comment  # noqa: E402
from schemas import Mood, RawComment  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.json"
METRICS_PATH = REPO_ROOT / "data" / "eval_metrics.json"

MAX_CONCURRENCY = 2      # max simultaneous LLM calls (free-tier safety)
STAGGER_SECONDS = 4.0    # gap between task starts (~15 requests/min)
LABELS = [m.value for m in Mood]


@dataclass
class CaseResult:
    index: int
    expected: str
    predicted: str | None
    error: str | None


def load_dataset() -> list[dict]:
    """Load the dataset and fail fast if any expected_mood drifts from the enum."""
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    valid = {m.value for m in Mood}
    bad = [
        (i, c.get("expected_mood"))
        for i, c in enumerate(cases)
        if c.get("expected_mood") not in valid
    ]
    if bad:
        raise SystemExit(f"Invalid expected_mood values in {DATASET_PATH}: {bad}")
    return cases


async def evaluate_case(
    case: dict, index: int, total: int, sem: asyncio.Semaphore
) -> CaseResult:
    """Analyse one dataset row under the shared concurrency cap."""
    async with sem:
        comment = RawComment(
            id=f"eval-{index:03d}",
            platform=case["platform"],
            text=case["text"],
            author="eval_harness",
            timestamp=datetime.now(timezone.utc),
        )
        try:
            analyzed = await analyze_comment(comment)
            result = CaseResult(
                index=index, expected=case["expected_mood"],
                predicted=analyzed.mood.value, error=None,
            )
        except Exception as exc:  # both providers failed — record and move on
            result = CaseResult(
                index=index, expected=case["expected_mood"],
                predicted=None, error=f"{type(exc).__name__}: {exc}",
            )
        status = "OK  " if result.error is None else "FAIL"
        pred = result.predicted or "-"
        print(f"[{index + 1:>2}/{total}] {status}  "
              f"expected={result.expected:<8} predicted={pred}")
        return result


async def run_benchmark() -> None:
    cases = load_dataset()
    print("=== Node 5: Mood Detection Benchmark ===")
    print(f"Loaded {len(cases)} cases from {DATASET_PATH}")
    print(f"Concurrency={MAX_CONCURRENCY}, stagger={STAGGER_SECONDS}s\n")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks: list[asyncio.Task[CaseResult]] = []
    for i, case in enumerate(cases):
        if i > 0:
            await asyncio.sleep(STAGGER_SECONDS)
        tasks.append(asyncio.create_task(evaluate_case(case, i, len(cases), sem)))
    results = await asyncio.gather(*tasks)

    succeeded = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    metrics: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(cases),
        "failed_cases": len(failed),
        "labels": LABELS,
    }

    if not succeeded:
        print("\nAll cases failed — check API keys / connectivity. No metrics written.")
        metrics.update({"accuracy": None, "confusion_matrix": None,
                        "classification_report": None})
    else:
        y_true = [r.expected for r in succeeded]
        y_pred = [r.predicted for r in succeeded]

        accuracy = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        report_str = classification_report(
            y_true, y_pred, labels=LABELS, digits=3, zero_division=0
        )
        report_dict = classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        )

        print("\n--- Classification Report ---")
        print(report_str)
        print(f"Confusion matrix (rows=true, cols=predicted): {LABELS}")
        for label, row in zip(LABELS, cm.tolist()):
            print(f"  {label:<9}{row}")
        print(f"\nAccuracy: {accuracy:.3f} "
              f"({len(succeeded)}/{len(cases)} scored, {len(failed)} failed)")

        metrics.update({
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),       # numpy -> native lists
            "classification_report": report_dict,  # per-mood precision/recall/F1
        })

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
