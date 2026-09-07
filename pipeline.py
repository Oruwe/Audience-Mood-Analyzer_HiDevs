"""Central streaming runner: source -> bounded queue -> rate-limited workers -> DuckDB.

Usage:
    python pipeline.py --mode mock
    python pipeline.py --mode live --keywords ai tech marketing
"""

import argparse
import asyncio
import logging
import time

from dotenv import load_dotenv

load_dotenv()  # MUST run before engine import: Router reads API keys at import time

from engine.anomaly_detector import detect_anomalies  # noqa: E402
from engine.embedder import generate_embedding  # noqa: E402
from engine.llm_client import analyze_comment, flush_observability  # noqa: E402
from ingestion.bluesky_stream import DEFAULT_KEYWORDS  # noqa: E402
from ingestion.broker import stream_inbound_comments  # noqa: E402
from schemas import RawComment  # noqa: E402
from storage.db import ainsert_enriched_record, close_connection  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")

_ALERT_COOLDOWN_SEC = 300.0  # minimum gap between logged crisis alerts


class TokenBucket:
    """Async token bucket — caps LLM throughput to protect free-tier quotas."""

    def __init__(self, rate_per_sec: float, burst: int = 1) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = rate_per_sec
        self.capacity = max(burst, 1)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
            await asyncio.sleep(deficit / self.rate)  # sleep OUTSIDE the lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Social-listening streaming pipeline")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent analysis workers (2-4 recommended)")
    parser.add_argument("--queue-size", type=int, default=100, help="Bounded buffer capacity")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Max LLM calls per minute (token bucket)")
    parser.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS,
                        help="Live-mode keyword filter")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    queue: asyncio.Queue[RawComment] = asyncio.Queue(maxsize=args.queue_size)
    bucket = TokenBucket(rate_per_sec=args.rate / 60.0)
    stop = asyncio.Event()

    source_mode = "bluesky" if args.mode == "live" else "mock"
    if args.mode == "mock":
        logger.info("Mode=mock | workers=%d queue=%d rate=%.0f/min",
                    args.workers, args.queue_size, args.rate)
    else:
        logger.info("Mode=live | keywords=%s | workers=%d rate=%.0f/min",
                    args.keywords, args.workers, args.rate)
    source = stream_inbound_comments(source_mode=source_mode, keywords=args.keywords)

    async def producer() -> None:
        async for comment in source:
            if stop.is_set():
                break
            await queue.put(comment)  # blocks when full -> natural backpressure

    async def worker(worker_id: int) -> None:
        while True:
            try:
                comment = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                if stop.is_set() and queue.empty():
                    logger.info("worker-%d exiting (queue drained)", worker_id)
                    return
                continue
            try:
                await bucket.acquire()
                record = await analyze_comment(comment)
                # Phase 4 theme clustering needs vectors. The embedder never
                # raises (local hash fallback), so this cannot kill the worker;
                # it shares the worker's rate slot to keep quota math simple.
                record.embedding = await generate_embedding(comment.text)
                await ainsert_enriched_record(record)
                logger.info(
                    "worker-%d | %s | %s | sentiment=%s intent=%s urgency=%.2f action=%s",
                    worker_id, comment.platform, comment.id,
                    record.sentiment.value, record.primary_intent.value,
                    record.urgency_score, record.recommended_action.value,
                )
            except Exception:
                logger.exception("worker-%d failed on comment %s", worker_id, comment.id)
            finally:
                queue.task_done()

    async def crisis_monitor() -> None:
        last_alert_at: float | None = None
        while not stop.is_set():
            await asyncio.sleep(60)
            try:
                alert = await detect_anomalies()
            except Exception:
                logger.exception("crisis_monitor iteration failed")
                continue
            if alert is None:
                continue
            now = time.monotonic()
            if last_alert_at is not None and (now - last_alert_at) < _ALERT_COOLDOWN_SEC:
                logger.info(
                    "crisis_monitor: %s alert suppressed (%.0fs of cooldown left)",
                    alert.severity, _ALERT_COOLDOWN_SEC - (now - last_alert_at),
                )
                continue
            last_alert_at = now
            logger.error("🚨 CRISIS ALERT: %s | %s", alert.theme, alert.trigger_reason)

    producer_task = asyncio.create_task(producer(), name="producer")
    worker_tasks = [asyncio.create_task(worker(i), name=f"worker-{i}") for i in range(args.workers)]
    monitor_task = asyncio.create_task(crisis_monitor(), name="crisis_monitor")

    try:
        await producer_task
    except asyncio.CancelledError:
        logger.info("Interrupt received — draining %d queued item(s)...", queue.qsize())
    finally:
        stop.set()
        if not producer_task.done():
            producer_task.cancel()
        await asyncio.gather(producer_task, return_exceptions=True)
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        await queue.join()                   # workers finish in-flight + queued items
        await asyncio.gather(*worker_tasks)  # workers exit once drained
        close_connection()
        flush_observability()
        logger.info("Shutdown complete. DuckDB connection closed.")


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass  # second Ctrl+C force-quits; first is handled gracefully above


if __name__ == "__main__":
    main()
