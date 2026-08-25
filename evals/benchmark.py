"""Node 5 — Evaluation layer.

Runs every case in evals/test_dataset.json through the production router
(engine.llm_client.analyze_comment), scores sentiment detection with a lenient
3-class scheme (strict accuracy on positive/neutral/negative; 'mixed' ground
truth counts as correct when the model predicts either polarity), prints a
classification report, and persists raw metrics to data/eval_metrics.json for
the Streamlit dashboard.

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
from schemas import RawComment  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "test_dataset.json"
METRICS_PATH = REPO_ROOT / "data" / "eval_metrics.json"

MAX_CONCURRENCY = 2      # max simultaneous LLM calls (free-tier safety)
STAGGER_SECONDS = 4.0    # gap between task starts (~15 requests/min)
COARSE_LABELS = ["positive", "neutral", "negative"]            # 3-class eval space
VALID_EXPECTED = {"positive", "neutral", "negative", "mixed"}  # dataset label set


@dataclass
class CaseResult:
    index: int
    expected: str
    predicted: str | None
    error: str | None


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


async def evaluate_case(
    case: dict, index: int, total: int, sem: asyncio.Semaphore
) -> CaseResult:
    """Analyse one dataset row under the shared concurrency cap."""
    async with sem:
        comment = RawComment(
            id=f"eval-{index:03d}",
            platform=case["platform"],
            text=case["text"],
            author_id="eval_harness",
            timestamp=datetime.now(timezone.utc),
        )
        try:
            analyzed = await analyze_comment(comment)
            result = CaseResult(
                index=index, expected=case["expected_mood"],
                predicted=analyzed.sentiment.value, error=None,
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
    print("=== Node 5: Sentiment Detection Benchmark ===")
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
        "labels": COARSE_LABELS,
    }

    if not succeeded:
        print("\nAll cases failed — check API keys / connectivity. No metrics written.")
        metrics.update({"accuracy": None, "accuracy_lenient": None,
                        "mixed_cases": 0, "mixed_covered": 0,
                        "confusion_matrix": None, "classification_report": None})
    else:
        def coarse(s: str) -> str:
            return {"strongly_positive": "positive",
                    "critical_escalation": "negative"}.get(s, s)

        scored = [(r.expected, coarse(r.predicted)) for r in succeeded]
        strict_pairs = [(e, p) for e, p in scored if e != "mixed"]
        mixed_preds = [p for e, p in scored if e == "mixed"]

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
        accuracy_lenient = (strict_correct + mixed_correct) / len(succeeded)

        if accuracy_strict is not None:
            print(f"\nStrict accuracy (non-mixed): {accuracy_strict:.3f} "
                  f"({len(strict_pairs)} scored)")
        else:
            print("\nStrict accuracy: n/a — every scored case was 'mixed'")
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

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
