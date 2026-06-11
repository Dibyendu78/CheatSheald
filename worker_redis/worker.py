#!/usr/bin/env python3
"""Redis-based ML Worker — Anti-cheating system (stateless).

Architecture:
    Redis frame_queue → Worker → ML inference → Redis result_channel (pub/sub)

Design:
    Fully stateless. Each frame is processed independently.
    No memory between frames — no counters, no history.
    Multiple workers can handle any user's frames in any order safely.

Usage:
    python worker.py
    docker compose up --scale worker=4
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import redis

# Ensure models directory on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_engine import ModelEngine

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] worker: %(message)s",
)
logger = logging.getLogger("worker")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
FRAME_QUEUE         = "frame_queue"
RESULT_CHANNEL_BASE = "result_channel"   # publish to result_channel:{api_instance_id}
USER_ROUTE_PREFIX   = "user_route"        # Redis key: user_route:{user_id} → api_instance_id
QUEUE_BLOCK_TIMEOUT = 5  # seconds to block on brpop

# How often to log queue size for monitoring (seconds)
MONITOR_INTERVAL       = int(os.environ.get("MONITOR_INTERVAL", "30"))

# Fix #2: Inference timeout — kill a hung ML call after N seconds
# Uses SIGALRM (Linux only). Set 0 to disable.
INFERENCE_TIMEOUT_SEC  = int(os.environ.get("INFERENCE_TIMEOUT_SEC", "10"))

# Fix #3: Max base64 frame size (~800 KB = ~600 KB raw JPEG at 1 FPS / 640p)
MAX_FRAME_B64_BYTES    = int(os.environ.get("MAX_FRAME_B64_BYTES", "819200"))  # 800 KB

# Redis startup retry config
REDIS_CONNECT_RETRIES = int(os.environ.get("REDIS_CONNECT_RETRIES", "10"))
REDIS_RETRY_BASE_DELAY = float(os.environ.get("REDIS_RETRY_BASE_DELAY", "2.0"))  # seconds


def _connect_redis() -> redis.Redis:
    """
    Connect to Redis with exponential backoff retry.

    Useful when the worker container starts before ElastiCache is reachable
    (e.g. EC2 cold start, VPC DNS propagation delay).

    Retry schedule (base=2s):
      attempt 1 → wait 2s
      attempt 2 → wait 4s
      attempt 3 → wait 8s  ... up to REDIS_CONNECT_RETRIES
    """
    for attempt in range(1, REDIS_CONNECT_RETRIES + 1):
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            client.ping()   # raises if not reachable
            logger.info(f"Redis connected on attempt {attempt} ({REDIS_HOST}:{REDIS_PORT})")
            return client
        except redis.RedisError as e:
            wait = REDIS_RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2, 4, 8, 16 ...
            if attempt == REDIS_CONNECT_RETRIES:
                logger.critical(
                    f"Redis unreachable after {REDIS_CONNECT_RETRIES} attempts. "
                    f"Last error: {e}. Giving up."
                )
                raise SystemExit(1)
            logger.warning(
                f"Redis not ready (attempt {attempt}/{REDIS_CONNECT_RETRIES}): {e}. "
                f"Retrying in {wait:.0f}s..."
            )
            time.sleep(wait)


class ProctoringWorker:
    """
    Fully STATELESS ML worker.

    Pipeline per frame:
        Redis BRPOP → decode → ML inference → Redis PUBLISH

    No memory between frames. Every frame is an independent decision.
    Multiple workers can process any user's frames in any order — safely.
    """

    def __init__(self) -> None:
        self.running = True
        self.processed = 0
        self.errors = 0

        # Redis connection — retries with exponential backoff until reachable
        self.r = _connect_redis()

        # Fix #1: In-process route cache — avoids Redis GET on every frame
        # Key: user_id → api_instance_id string
        # Invalidated when user disconnects (route key deleted from Redis)
        self._route_cache: dict[str, str] = {}
        self._cache_hits   = 0
        self._cache_misses = 0

        # ML model engine — loaded once per worker process
        models_dir = Path(os.environ.get("MODELS_DIR", "/app"))
        logger.info(f"Loading ML models from {models_dir}...")

        self.engine = ModelEngine(
            object_model_path=str(models_dir / "best.pt"),
            person_model_path=str(models_dir / "yolov8n.pt"),
            face_landmarker_path=str(models_dir / "face_landmarker.task"),
        )
        logger.info("ML models loaded successfully.")

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)

        # Fix #2: SIGALRM handler for inference timeout (Linux only)
        if INFERENCE_TIMEOUT_SEC > 0:
            signal.signal(signal.SIGALRM, self._inference_timeout_handler)

    def _shutdown(self, signum, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _inference_timeout_handler(self, signum, frame) -> None:
        """Fix #2: Called by SIGALRM when inference hangs past INFERENCE_TIMEOUT_SEC."""
        raise TimeoutError(f"ML inference exceeded {INFERENCE_TIMEOUT_SEC}s timeout")

    def _get_route(self, user_id: str) -> str | None:
        """
        Fix #1: Route cache — check local dict first, Redis only on miss.

        At 1000 users × 1 FPS:
          Without cache: 1000 Redis GET/sec
          With cache:    ~0 Redis GET/sec (after warmup)

        Cache is invalidated naturally: if a user reconnects to a different
        API instance, the Redis key is updated. We check Redis on miss,
        so the cache self-corrects within one frame after a reconnect.
        """
        target = self._route_cache.get(user_id)
        if target:
            self._cache_hits += 1
            return target

        # Cache miss — fetch from Redis and populate cache
        target = self.r.get(f"{USER_ROUTE_PREFIX}:{user_id}")
        self._cache_misses += 1
        if target:
            self._route_cache[user_id] = target
        return target

    def _publish(self, channel: str, payload: str) -> None:
        """
        Fix #4: Publish with one retry on Redis hiccup.
        Transient network blips to ElastiCache should not lose a result.
        """
        try:
            self.r.publish(channel, payload)
        except redis.RedisError as e:
            logger.warning(f"Publish failed ({e}), retrying once...")
            time.sleep(0.1)
            self.r.publish(channel, payload)  # raises if still failing → caught by caller

    def _process_frame(self, payload: dict) -> None:
        """
        Stateless frame processor.

        Flow: size check → route lookup (cached) → ML inference (timeout) → publish (retry)
        """
        user_id: str   = payload.get("user_id", "")
        frame_b64: str = payload.get("frame", "")

        if not user_id:
            logger.warning("Received frame with missing user_id — DISCARDING")
            return

        if not frame_b64:
            logger.warning(f"Empty frame for user_id={user_id} — DISCARDING")
            return

        # Fix #3: Frame size guard — reject oversized payloads before decoding
        frame_size = len(frame_b64)
        if frame_size > MAX_FRAME_B64_BYTES:
            logger.warning(
                f"Oversized frame for user={user_id}: "
                f"{frame_size} bytes > {MAX_FRAME_B64_BYTES} limit — DISCARDING"
            )
            return

        start_time = time.time()

        try:
            # Fix #2: Set SIGALRM before inference, cancel after
            if INFERENCE_TIMEOUT_SEC > 0:
                signal.alarm(INFERENCE_TIMEOUT_SEC)
            try:
                ml_result = self.engine.analyze_frame_b64(
                    frame_b64=frame_b64,
                    user_id=user_id,
                )
            finally:
                if INFERENCE_TIMEOUT_SEC > 0:
                    signal.alarm(0)  # always cancel alarm

            elapsed_ms = (time.time() - start_time) * 1000
            ml_result["processing_latency_ms"] = round(elapsed_ms, 1)

            # Fix #1: Use cached route — avoids Redis GET on every frame
            target_api = self._get_route(user_id)
            if target_api is None:
                logger.debug(f"No route for user={user_id} — disconnected, discarding result")
                return

            target_channel = f"{RESULT_CHANNEL_BASE}:{target_api}"

            # Fix #4: Publish with one automatic retry
            self._publish(target_channel, json.dumps({
                "user_id": user_id,
                "result":  ml_result,
            }))

            self.processed += 1
            gaze_ok = ml_result.get("details", {}).get("gaze_ok", True)
            status  = "[CHEATING]" if ml_result.get("cheating") else "[OK]"
            logger.info(
                f"{status} user={user_id} | "
                f"→ {target_channel} | "
                f"gaze={'OK' if gaze_ok else 'AWAY'} | "
                f"faces={ml_result['details'].get('face_count', 0)} | "
                f"latency={elapsed_ms:.0f}ms | processed={self.processed}"
            )

        except TimeoutError as e:
            self.errors += 1
            if INFERENCE_TIMEOUT_SEC > 0:
                signal.alarm(0)   # ensure alarm is cleared
            logger.error(f"Inference timeout for user={user_id}: {e}")

        except Exception as e:
            self.errors += 1
            logger.error(f"Error processing frame for user_id={user_id}: {e}", exc_info=True)
            # Notify user even on error (with retry)
            try:
                target_api = self._get_route(user_id)
                if target_api:
                    self._publish(f"{RESULT_CHANNEL_BASE}:{target_api}", json.dumps({
                        "user_id": user_id,
                        "result": {
                            "cheating": False,
                            "type":    "error",
                            "message": "Processing error — frame skipped",
                            "user_id": user_id,
                            "details": {},
                            "alerts":  [],
                        },
                    }))
            except Exception:
                pass

    def run(self) -> None:
        """Main worker loop — blocks on Redis BRPOP, processes frames."""
        logger.info(f"Worker started (stateless mode). Listening on: {FRAME_QUEUE}")
        logger.info(f"Routing: {USER_ROUTE_PREFIX}:{{user_id}} → {RESULT_CHANNEL_BASE}:{{api_instance}}")

        last_monitor = time.time()

        while self.running:
            try:
                item = self.r.brpop(FRAME_QUEUE, timeout=QUEUE_BLOCK_TIMEOUT)

                if item is None:
                    # Periodic queue monitoring on idle timeouts
                    if time.time() - last_monitor >= MONITOR_INTERVAL:
                        queue_size = self.r.llen(FRAME_QUEUE)
                        logger.info(
                            f"[MONITOR] queue={queue_size} | "
                            f"processed={self.processed} | errors={self.errors}"
                        )
                        last_monitor = time.time()
                    continue

                _, raw_payload = item
                payload = json.loads(raw_payload)
                self._process_frame(payload)

                # Periodic monitoring after every N frames — includes cache stats
                if self.processed % 50 == 0 and self.processed > 0:
                    queue_size  = self.r.llen(FRAME_QUEUE)
                    total_lookups = self._cache_hits + self._cache_misses
                    hit_rate = (
                        f"{100 * self._cache_hits / total_lookups:.1f}%"
                        if total_lookups else "n/a"
                    )
                    logger.info(
                        f"[MONITOR] queue={queue_size} | "
                        f"processed={self.processed} | errors={self.errors} | "
                        f"route_cache_hit={hit_rate}"
                    )
                    last_monitor = time.time()

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode queue message: {e}")
                self.errors += 1
            except redis.RedisError as e:
                logger.error(f"Redis error: {e}")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Unexpected worker error: {e}", exc_info=True)
                self.errors += 1

        logger.info(f"Worker stopped. Processed={self.processed}, Errors={self.errors}")


def main() -> None:
    worker = ProctoringWorker()
    worker.run()


if __name__ == "__main__":
    main()
