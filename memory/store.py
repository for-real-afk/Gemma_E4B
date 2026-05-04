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
import logging
import time
from threading import Lock

from .models import SessionMemory

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS:    int = 60 * 60      # 1 hour of inactivity → evict
PRUNE_INTERVAL_SECONDS: int = 60 * 10      # run pruner every 10 min
# ─────────────────────────────────────────────────────────────────────────────

_sessions:      dict[str, SessionMemory] = {}
_last_active:   dict[str, float]         = {}
_lock:          Lock                     = Lock()


# ── public API ────────────────────────────────────────────────────────────────

def get_or_create(session_id: str) -> SessionMemory:
    """Return existing session or create a fresh one."""
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id]    = SessionMemory(session_id=session_id)
            logger.info("Created session %s", session_id)
        _last_active[session_id] = time.monotonic()
        return _sessions[session_id]


def save(session: SessionMemory) -> None:
    """Persist updated session state back to the store."""
    with _lock:
        _sessions[session.session_id]    = session
        _last_active[session.session_id] = time.monotonic()


def delete(session_id: str) -> None:
    """Explicitly remove a session (e.g. user logout)."""
    with _lock:
        _sessions.pop(session_id, None)
        _last_active.pop(session_id, None)
        logger.info("Deleted session %s", session_id)


def active_session_count() -> int:
    with _lock:
        return len(_sessions)


# ── background pruner ─────────────────────────────────────────────────────────

async def prune_stale_sessions() -> None:
    """
    Async background task: evict sessions idle longer than SESSION_TTL_SECONDS.
    Register via FastAPI lifespan (see main.py).
    """
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        cutoff = time.monotonic() - SESSION_TTL_SECONDS
        with _lock:
            stale = [sid for sid, t in _last_active.items() if t < cutoff]
            for sid in stale:
                _sessions.pop(sid, None)
                _last_active.pop(sid, None)
        if stale:
            logger.info("Pruned %d stale sessions: %s", len(stale), stale)