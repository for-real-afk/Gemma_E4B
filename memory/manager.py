"""
memory/manager.py
──────────────────
Orchestrates the two-layer memory system:

  Layer 1 — Short-term  : last SHORT_TERM_TURNS turns (verbatim, authoritative)
  Layer 2 — Summary     : compressed older context   (background only)

Prompt assembly order (critical — never reorder):
  ┌─────────────────────────────────┐
  │  1. SYSTEM PROMPT               │
  │  2. SUMMARY MEMORY (background) │
  │  3. RECENT TURNS  (authoritative)│
  └─────────────────────────────────┘

Multimodal contract:
  - Raw images are NEVER stored.
  - Images are converted to a text `media_description` BEFORE calling add_turn().
  - `media_description` lives verbatim in short-term; extracted facts go into summary.

Summarization is NOT on the hot path.
  - add_turn_and_get_prompt() returns immediately after appending the turn.
  - It returns a (messages, needs_summary, overflow) tuple so the caller
    can schedule summarize_in_background() as a FastAPI BackgroundTask —
    running after the HTTP response is already sent to the user.
"""

import logging
from typing import Callable, Awaitable, Optional

from .models     import Role, SessionMemory, Turn
from .store      import get_or_create, save
from .summarizer import make_summary

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
SHORT_TERM_TURNS: int = 5   # turns kept verbatim (each turn = 1 user + 1 assistant msg)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a helpful, accurate assistant. always respond short and concise unless the user explicitly asks for a detailed answer.  If you don't know the answer, say you don't know.  Never try to make up an answer.


Memory rules (strictly follow):
1. The RECENT CONVERSATION below is authoritative — trust it over everything else.
2. The SUMMARY MEMORY is background context only — use it to understand intent \
and past decisions.
3. If the summary conflicts with recent messages, ignore the summary.
4. Never claim to remember an image unless its description appears in the \
recent conversation below.
"""


# ── public API ────────────────────────────────────────────────────────────────

def add_turn_and_get_prompt(
    session_id:        str,
    user_message:      str,
    media_description: Optional[str] = None,
) -> tuple[list[dict], list[Turn], str]:
    """
    Synchronous and fast — zero LLM calls, zero blocking I/O.

    1. Load (or create) the session.
    2. Append the new user turn.
    3. Split short_term if it exceeds the window (overflow is returned,
       NOT summarized yet — that happens in the background).
    4. Return the assembled prompt immediately.

    Parameters
    ----------
    session_id        : unique identifier for this conversation
    user_message      : the raw text the user typed
    media_description : text-semantic extracted from any uploaded image/media
                        (pass None for text-only turns)

    Returns
    -------
    messages  : list[dict]  — [{role, content}, …] ready for the LLM
    overflow  : list[Turn]  — older turns evicted from short_term this call
                              (empty list if no eviction happened)
    old_summary : str       — the summary value at the time of eviction,
                              needed by summarize_in_background()
    """
    session = get_or_create(session_id)

    session.short_term.append(Turn(
        role              = Role.USER,
        content           = user_message,
        media_description = media_description,
    ))
    session.turn_count += 1

    # Evict overflow synchronously (just a list slice — instant)
    overflow, old_summary = _evict_overflow(session)

    save(session)
    return _assemble_prompt(session), overflow, old_summary


def record_assistant_reply(session_id: str, reply: str) -> None:
    """
    Call this AFTER the LLM responds to store the assistant turn.
    Keeping this separate lets the caller decide whether to store
    error responses or not.
    """
    session = get_or_create(session_id)
    session.short_term.append(Turn(role=Role.ASSISTANT, content=reply))
    session.turn_count += 1
    save(session)


async def summarize_in_background(
    session_id:  str,
    overflow:    list[Turn],
    old_summary: str,
    llm_caller:  Callable[[str], Awaitable[dict]],
) -> None:
    """
    Background coroutine — runs AFTER the HTTP response is sent.

    Compress overflow turns into the session summary without blocking
    any user-facing request.  Safe to call with an empty overflow list
    (returns immediately).

    Register via FastAPI BackgroundTasks:
        background_tasks.add_task(
            summarize_in_background,
            session_id, overflow, old_summary, call_llm
        )
    """
    if not overflow:
        return

    logger.info(
        "Session %s: background summarizing %d evicted messages.",
        session_id, len(overflow),
    )

    new_summary = await make_summary(
        turns            = overflow,
        existing_summary = old_summary,
        llm_caller       = llm_caller,
    )

    # Write the new summary back — short_term was already saved in add_turn_and_get_prompt
    session = get_or_create(session_id)
    session.summary = new_summary
    save(session)
    logger.info("Session %s: summary updated (%d chars).", session_id, len(new_summary))


def get_session_stats(session_id: str) -> dict:
    """Lightweight introspection — useful for the /health endpoint or debugging."""
    session = get_or_create(session_id)
    return {
        "session_id":       session_id,
        "total_turns":      session.turn_count,
        "short_term_turns": len(session.short_term),
        "summary_chars":    len(session.summary),
        "has_summary":      bool(session.summary),
    }


# ── private helpers ───────────────────────────────────────────────────────────

def _evict_overflow(session: SessionMemory) -> tuple[list[Turn], str]:
    """
    Synchronously split short_term when it exceeds the window.
    Returns (overflow_turns, old_summary) so the caller can hand them
    to summarize_in_background() without another store lookup.
    """
    max_messages = SHORT_TERM_TURNS * 2  # user + assistant per turn

    if len(session.short_term) <= max_messages:
        return [], session.summary   # nothing to evict

    overflow           = session.short_term[:-max_messages]
    session.short_term = session.short_term[-max_messages:]
    old_summary        = session.summary   # snapshot before background updates it

    logger.info(
        "Session %s: evicted %d messages from short_term for background summarization.",
        session.session_id, len(overflow),
    )
    return overflow, old_summary


def _assemble_prompt(session: SessionMemory) -> list[dict]:
    """
    Build the final messages list in the required order:

      system  →  summary (injected as system context)  →  recent turns
    """
    messages: list[dict] = []

    # 1. System prompt
    messages.append({"role": "system", "content": _SYSTEM_PROMPT})

    # 2. Summary memory (background) — injected as a system-level context block
    #    so it never outranks the recent turns in the LLM's attention.
    if session.summary:
        summary_block = (
            "[SUMMARY MEMORY — background context only, lower priority than recent messages]\n"
            + session.summary
        )
        messages.append({"role": "system", "content": summary_block})

    # 3. Recent turns (authoritative)
    for turn in session.short_term:
        messages.append(turn.to_llm_message())

    return messages