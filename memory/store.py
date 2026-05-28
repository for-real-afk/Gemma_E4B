"""
memory/store.py
───────────────
Thread-safe in-memory session store.

Lifecycle:
  - Sessions are created on first message.
  - Sessions expire after SESSION_TTL_SECONDS of inactivity.
  - A background task (started with the FastAPI lifespan) prunes stale sessions.

Replace the `_sessions` dict + `_lock` with a Redis client here
if you ever need multi-process persistence — the rest of the code
won't need to change.
"""

import asyncio
import json
import logging
import os
import time
from threading import Lock

from .models import SessionMemory, Turn, Role

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS:    int = 60 * 60      # 1 hour of inactivity → evict
PRUNE_INTERVAL_SECONDS: int = 60 * 10      # run pruner every 10 min
# ─────────────────────────────────────────────────────────────────────────────

_sessions:      dict[str, SessionMemory] = {}
_last_active:   dict[str, float]         = {}
_lock:          Lock                     = Lock()

# ── Cache Configuration ───────────────────────────────────────────────────────
CACHE_TYPE = os.getenv("CACHE_TYPE", "in_memory").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client = None
if CACHE_TYPE == "redis":
    try:
        import redis
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        # Test connection
        _redis_client.ping()
        logger.info("Successfully connected to Redis cache at %s", REDIS_URL)
    except Exception as e:
        logger.error("Failed to connect to Redis, falling back to in-memory cache: %s", e)
        CACHE_TYPE = "in_memory"
        _redis_client = None


# ── serialization helpers ─────────────────────────────────────────────────────

def _serialize(session: SessionMemory) -> str:
    return json.dumps({
        "session_id": session.session_id,
        "short_term": [
            {
                "role": turn.role.value,
                "content": turn.content,
                "media_description": turn.media_description
            }
            for turn in session.short_term
        ],
        "summary": session.summary,
        "turn_count": session.turn_count
    })


def _deserialize(data_str: str) -> SessionMemory:
    data = json.loads(data_str)
    short_term = [
        Turn(
            role=Role(t["role"]),
            content=t["content"],
            media_description=t.get("media_description")
        )
        for t in data.get("short_term", [])
    ]
    return SessionMemory(
        session_id=data["session_id"],
        short_term=short_term,
        summary=data.get("summary", ""),
        turn_count=data.get("turn_count", 0)
    )


# ── public API ────────────────────────────────────────────────────────────────

def get_or_create(session_id: str) -> SessionMemory:
    """Return existing session or create a fresh one."""
    global CACHE_TYPE, _redis_client
    if CACHE_TYPE == "redis" and _redis_client:
        try:
            key = f"session:{session_id}"
            data = _redis_client.get(key)
            if data:
                _redis_client.expire(key, SESSION_TTL_SECONDS)
                return _deserialize(data)
            # Create fresh
            session = SessionMemory(session_id=session_id)
            _redis_client.setex(key, SESSION_TTL_SECONDS, _serialize(session))
            logger.info("Created session %s in Redis", session_id)
            return session
        except Exception as e:
            logger.error("Redis get_or_create failed, falling back to in-memory: %s", e)
            # Temporarily fall back to in-memory, but do not override CACHE_TYPE permanently
            # so that we can attempt reconnection on subsequent requests if needed

    # Fallback to in-memory
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id]    = SessionMemory(session_id=session_id)
            logger.info("Created session %s", session_id)
        _last_active[session_id] = time.monotonic()
        return _sessions[session_id]


def save(session: SessionMemory) -> None:
    """Persist updated session state back to the store."""
    if CACHE_TYPE == "redis" and _redis_client:
        try:
            key = f"session:{session.session_id}"
            _redis_client.setex(key, SESSION_TTL_SECONDS, _serialize(session))
            return
        except Exception as e:
            logger.error("Redis save failed, falling back to in-memory: %s", e)

    # Fallback to in-memory
    with _lock:
        _sessions[session.session_id]    = session
        _last_active[session.session_id] = time.monotonic()


def delete(session_id: str) -> None:
    """Explicitly remove a session (e.g. user logout)."""
    if CACHE_TYPE == "redis" and _redis_client:
        try:
            key = f"session:{session_id}"
            _redis_client.delete(key)
            logger.info("Deleted session %s", session_id)
            return
        except Exception as e:
            logger.error("Redis delete failed, falling back to in-memory: %s", e)

    # Fallback to in-memory
    with _lock:
        _sessions.pop(session_id, None)
        _last_active.pop(session_id, None)
        logger.info("Deleted session %s", session_id)


def active_session_count() -> int:
    if CACHE_TYPE == "redis" and _redis_client:
        try:
            count = 0
            for _ in _redis_client.scan_iter("session:*"):
                count += 1
            return count
        except Exception as e:
            logger.error("Redis active_session_count failed, falling back to in-memory: %s", e)

    with _lock:
        return len(_sessions)


# ── background pruner ─────────────────────────────────────────────────────────

async def prune_stale_sessions() -> None:
    """
    Async background task: evict sessions idle longer than SESSION_TTL_SECONDS.
    Only active for in-memory cache; Redis handles TTL natively.
    """
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        
        # If using Redis, bypass active pruning
        if CACHE_TYPE == "redis" and _redis_client:
            continue
            
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        with _lock:
            stale = [sid for sid, t in _last_active.items() if t < cutoff]
            for sid in stale:
                _sessions.pop(sid, None)
                _last_active.pop(sid, None)
        if stale:
            logger.info("Pruned %d stale sessions: %s", len(stale), stale)