#!/usr/bin/env python3
"""FastAPI WebSocket server — Multi-instance anti-cheating system (v3).

Upgrades over v2:
  1. Instance-aware routing   — workers publish to THIS instance's channel only.
                                No broadcasting to all API instances.
  2. Backpressure             — frames dropped + client notified when queue is full.
  3. Monitoring               — /metrics exposes queue size, active users, health.

Architecture:
    Frontend → ALB (sticky sessions) → API instance N
                                          ↓
                                    Redis frame_queue
                                          ↓
                                       Worker
                                          ↓ lookup user_route:{user_id} → "api-N"
                                    result_channel:api-N  (direct, no broadcast)
                                          ↓
                                    API instance N → WebSocket → user
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from typing import Dict

from dotenv import load_dotenv

# Load .env file when running locally (no-op inside Docker where env is injected)
load_dotenv()

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

# ── Config ──────────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

FRAME_QUEUE         = "frame_queue"
RESULT_CHANNEL_BASE = "result_channel"    # workers publish to result_channel:{instance_id}
USER_ROUTE_PREFIX   = "user_route"        # Redis key: user_route:{user_id} → instance_id
USER_ROUTE_TTL      = 7200               # 2 hrs — matches ALB sticky session duration

# Backpressure: max pending frames before new ones are dropped
BACKPRESSURE_LIMIT  = int(os.environ.get("BACKPRESSURE_LIMIT", "500"))

# Per-user rate limit: max frames per second accepted from a single user.
# Prevents one heavy sender from flooding the queue and starving other users.
MAX_FPS_PER_USER    = int(os.environ.get("MAX_FPS_PER_USER", "3"))
RATE_LIMIT_PREFIX   = "rate_limit"   # Redis key: rate_limit:{user_id} (TTL = 1s window)

# ── Instance identity ────────────────────────────────────────────────────────
# Docker assigns a unique hostname per scaled container (e.g. "testing-api-1").
# Workers use this to route results directly here — no broadcast to all instances.
API_INSTANCE_ID  = os.environ.get("API_INSTANCE_ID", socket.gethostname())
MY_RESULT_CHANNEL = f"{RESULT_CHANNEL_BASE}:{API_INSTANCE_ID}"

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Anti-Cheating API",
    description=f"Instance: {API_INSTANCE_ID}",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Per-instance state ────────────────────────────────────────────────────────
# ONLY connections owned by THIS instance. Never shared across instances.
active_connections: Dict[str, WebSocket] = {}

# ── Redis ─────────────────────────────────────────────────────────────────────
redis_client: aioredis.Redis | None = None

# ── Metrics (in-memory, per-instance) ─────────────────────────────────────────
_stats = {
    "frames_received": 0,
    "frames_dropped_backpressure": 0,
    "results_delivered": 0,
    "results_missed": 0,       # result arrived but user not on this instance
    "started_at": time.time(),
}


@app.on_event("startup")
async def startup_event() -> None:
    global redis_client
    logger.info(f"═══ API instance starting: {API_INSTANCE_ID} ═══")
    logger.info(f"Result channel  : {MY_RESULT_CHANNEL}")
    logger.info(f"Backpressure    : {BACKPRESSURE_LIMIT} frames max")

    # Connect to Redis with exponential backoff retry
    # Handles EC2 cold starts where ElastiCache DNS may not resolve immediately
    max_retries   = 10
    base_delay    = 2.0  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            redis_client = aioredis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await redis_client.ping()
            logger.info(f"Redis connected on attempt {attempt} ({REDIS_HOST}:{REDIS_PORT})")
            break
        except Exception as e:
            wait = base_delay * (2 ** (attempt - 1))  # 2, 4, 8, 16 ...
            if attempt == max_retries:
                logger.critical(f"Redis unreachable after {max_retries} attempts: {e}")
                raise RuntimeError(f"Cannot connect to Redis: {e}") from e
            logger.warning(
                f"Redis not ready (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {wait:.0f}s..."
            )
            await asyncio.sleep(wait)

    asyncio.create_task(result_listener())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if redis_client:
        await redis_client.close()
    logger.info(f"API instance {API_INSTANCE_ID} shut down.")


# ── Result Router ─────────────────────────────────────────────────────────────
async def result_listener() -> None:
    """
    Subscribe ONLY to this instance's channel: result_channel:{API_INSTANCE_ID}

    Workers look up user_route:{user_id} → API_INSTANCE_ID, then publish
    to result_channel:{API_INSTANCE_ID} directly.

    ✅ No broadcast — only this instance receives this message.
    ✅ Zero wasted CPU on other instances.
    """
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(MY_RESULT_CHANNEL)
    logger.info(f"Subscribed to: {MY_RESULT_CHANNEL}")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            payload = json.loads(message["data"])
            user_id = payload.get("user_id")
            result  = payload.get("result")

            if not user_id:
                continue

            ws = active_connections.get(user_id)
            if ws is None:
                # Should not happen if routing is correct — log but don't error
                _stats["results_missed"] += 1
                logger.warning(f"Result arrived for user={user_id} but no WS on this instance")
                continue

            await ws.send_text(json.dumps({
                "type":    "result",
                "user_id": user_id,
                "result":  result,
            }))
            _stats["results_delivered"] += 1

        except Exception as e:
            logger.error(f"result_listener error: {e}")


# ── WebSocket Endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Accept frames from a student client.

    Per-frame flow:
      1. Validate JSON {user_id, frame}
      2. Set Redis route FIRST (race-condition safe)   ← fix #1
      3. Register local WebSocket mapping
      4. Per-user rate limit check                     ← fix #4 (queue fairness)
      5. Backpressure check                            (drop if queue > limit)
      6. LPUSH frame to frame_queue                    (worker picks it up)
      7. ACK client immediately
    """
    await websocket.accept()
    user_id: str | None = None

    try:
        async for raw in websocket.iter_text():
            try:
                data      = json.loads(raw)
                user_id   = data.get("user_id")
                frame_b64 = data.get("frame")

                if not user_id:
                    await websocket.send_text(json.dumps({"type": "error", "message": "user_id is required"}))
                    continue
                if not frame_b64:
                    await websocket.send_text(json.dumps({"type": "error", "message": "frame is required"}))
                    continue

                # ── Fix #1: Redis route FIRST, then local dict ──────────────
                # If anything fails between these two lines, Redis is the
                # authoritative state. Worker can route; we just won't have
                # a local WS. The result_listener logs a miss instead of
                # silently routing to the wrong socket.
                await redis_client.set(
                    f"{USER_ROUTE_PREFIX}:{user_id}",
                    API_INSTANCE_ID,
                    ex=USER_ROUTE_TTL,   # TTL refreshed on EVERY frame ✅
                )
                active_connections[user_id] = websocket

                _stats["frames_received"] += 1

                # ── Fix #4: Per-user rate limiting (queue fairness) ──────────
                # Sliding 1-second window per user_id.
                # Prevents one student from flooding the queue at 30 FPS
                # and starving other users who send at 1 FPS.
                rate_key   = f"{RATE_LIMIT_PREFIX}:{user_id}"
                frame_count = await redis_client.incr(rate_key)
                if frame_count == 1:
                    # First frame this second — start the 1s TTL window
                    await redis_client.expire(rate_key, 1)
                if frame_count > MAX_FPS_PER_USER:
                    logger.debug(f"Rate limited: user={user_id} sent {frame_count} FPS > {MAX_FPS_PER_USER}")
                    await websocket.send_text(json.dumps({
                        "type":    "rate_limited",
                        "message": f"Too fast — max {MAX_FPS_PER_USER} FPS allowed.",
                        "fps_limit": MAX_FPS_PER_USER,
                    }))
                    continue

                # ── Backpressure ────────────────────────────────────────────
                queue_size = await redis_client.llen(FRAME_QUEUE)
                if queue_size > BACKPRESSURE_LIMIT:
                    _stats["frames_dropped_backpressure"] += 1
                    logger.warning(
                        f"BACKPRESSURE: queue={queue_size} > {BACKPRESSURE_LIMIT} "
                        f"| dropping frame for user={user_id}"
                    )
                    await websocket.send_text(json.dumps({
                        "type":       "backpressure",
                        "message":    "System overloaded — frame dropped. Reduce send rate.",
                        "queue_size": queue_size,
                        "limit":      BACKPRESSURE_LIMIT,
                    }))
                    continue

                # ── Enqueue frame ────────────────────────────────────────────
                await redis_client.lpush(FRAME_QUEUE, json.dumps({
                    "user_id": user_id,
                    "frame":   frame_b64,
                    "ts":      time.time(),
                }))

                await websocket.send_text(json.dumps({
                    "type":       "ack",
                    "user_id":    user_id,
                    "queue_size": queue_size + 1,
                }))
                logger.debug(f"Queued frame: user={user_id} queue={queue_size + 1}")

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))

    except WebSocketDisconnect:
        logger.info(f"Disconnected: user_id={user_id}")
    finally:
        if user_id:
            active_connections.pop(user_id, None)
            # Remove routing entry immediately — prevents stale routes
            try:
                await redis_client.delete(f"{USER_ROUTE_PREFIX}:{user_id}")
            except Exception:
                pass
            logger.info(f"Cleaned up user={user_id}. Active={len(active_connections)}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    redis_ok = False
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        pass
    return {
        "status":      "healthy" if redis_ok else "degraded",
        "instance_id": API_INSTANCE_ID,
        "redis":       redis_ok,
        "active_users": len(active_connections),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics_endpoint() -> dict:
    """
    Per-instance metrics. Each API instance exposes its own stats.
    Aggregate these across instances for full system view.
    """
    queue_size = 0
    try:
        queue_size = await redis_client.llen(FRAME_QUEUE)
    except Exception:
        pass

    uptime_s = round(time.time() - _stats["started_at"], 1)
    return {
        "instance_id":                 API_INSTANCE_ID,
        "result_channel":              MY_RESULT_CHANNEL,
        "uptime_seconds":              uptime_s,
        "active_connections":          len(active_connections),
        "connected_users":             list(active_connections.keys()),
        # Queue
        "frame_queue_size":            queue_size,
        "backpressure_limit":          BACKPRESSURE_LIMIT,
        "overloaded":                  queue_size > BACKPRESSURE_LIMIT,
        # Counters
        "frames_received":             _stats["frames_received"],
        "frames_dropped_backpressure": _stats["frames_dropped_backpressure"],
        "results_delivered":           _stats["results_delivered"],
        "results_missed":              _stats["results_missed"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
