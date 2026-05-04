"""
memory/
───────
Two-layer LLM memory system.

Public surface (import from here):
  add_turn_and_get_prompt  — call before each LLM request
  record_assistant_reply   — call after each LLM response
  get_session_stats        — lightweight introspection
  prune_stale_sessions     — async background task (register in lifespan)
"""

from .manager import add_turn_and_get_prompt, record_assistant_reply, get_session_stats, summarize_in_background
from .store   import prune_stale_sessions, active_session_count

__all__ = [
    "add_turn_and_get_prompt",
    "record_assistant_reply",
    "get_session_stats",
    "prune_stale_sessions",
    "active_session_count",
    "summarize_in_background",
]