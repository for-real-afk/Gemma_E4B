"""
memory/summarizer.py
─────────────────────
Two-layer summarization strategy:

  Primary  → call Ollama to compress old turns into a coherent summary
  Fallback → if the LLM call fails, concatenate key fields and hard-truncate

The summarizer is intentionally decoupled from the LLM caller in main.py
so it can be unit-tested independently.
"""

import logging
import json
from typing import Callable, Awaitable

from .models import Turn

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
SUMMARY_TARGET_CHARS: int = 1_200   # ~300 tokens at ~4 chars/token
SUMMARY_MAX_CHARS:    int = 1_500   # hard ceiling before truncation kicks in
# ─────────────────────────────────────────────────────────────────────────────

_SUMMARIZE_PROMPT = """\
You are a memory compression assistant.

Your job is to compress a conversation excerpt into a short, factual summary \
that a language model can use as background context.

Rules:
1. Capture only decisions, constraints, user preferences, and key facts.
2. Omit pleasantries, filler, and repeated information.
3. If images/media were described, store only the extracted facts — not \
"user shared an image".
4. Keep the summary under 300 tokens (roughly 1 200 characters).
5. Write in third-person neutral style.
6. Output ONLY the summary text — no preamble, no bullet points unless \
they genuinely help compress the content.

Conversation excerpt to compress:
{conversation}

Existing summary to merge with (may be empty):
{existing_summary}
"""


# ── public API ────────────────────────────────────────────────────────────────

async def make_summary(
    turns:            list[Turn],
    existing_summary: str,
    llm_caller:       Callable[[str], Awaitable[dict]],
) -> str:
    """
    Attempt LLM-based summarization; fall back to truncation on any error.

    Parameters
    ----------
    turns            : the older turns to compress (NOT the recent short-term window)
    existing_summary : the current running summary to merge into
    llm_caller       : coroutine that accepts a prompt string and returns
                       {"response": "<text>"} — pass `call_llm` from main.py
    """
    try:
        return await _llm_summary(turns, existing_summary, llm_caller)
    except Exception as exc:
        logger.warning("LLM summarizer failed (%s); using truncation fallback.", exc)
        return _truncation_fallback(turns, existing_summary)


# ── private helpers ───────────────────────────────────────────────────────────

async def _llm_summary(
    turns:            list[Turn],
    existing_summary: str,
    llm_caller:       Callable[[str], Awaitable[dict]],
) -> str:
    conversation_text = _turns_to_text(turns)
    prompt = _SUMMARIZE_PROMPT.format(
        conversation     = conversation_text,
        existing_summary = existing_summary or "(none)",
    )
    result  = await llm_caller(prompt)
    summary = result.get("response", "").strip()

    if not summary:
        raise ValueError("LLM returned empty summary")

    # Hard-cap to avoid runaway growth
    if len(summary) > SUMMARY_MAX_CHARS:
        summary = summary[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + " …"

    logger.debug("LLM summary produced (%d chars)", len(summary))
    return summary


def _truncation_fallback(turns: list[Turn], existing_summary: str) -> str:
    """
    Rule-based fallback: concatenate existing summary + new turns,
    then hard-truncate to SUMMARY_TARGET_CHARS from the END
    (we want to keep the most recent info).
    """
    parts: list[str] = []

    if existing_summary:
        parts.append(f"[Previous context]\n{existing_summary}")

    parts.append("[Recent turns]\n" + _turns_to_text(turns))

    combined = "\n\n".join(parts)

    if len(combined) <= SUMMARY_TARGET_CHARS:
        logger.debug("Truncation fallback: no trimming needed (%d chars)", len(combined))
        return combined

    # Keep the tail (most recent content is more valuable)
    truncated = "…" + combined[-SUMMARY_TARGET_CHARS:]
    logger.debug("Truncation fallback: trimmed to %d chars", len(truncated))
    return truncated


def _turns_to_text(turns: list[Turn]) -> str:
    lines: list[str] = []
    for t in turns:
        prefix = "User" if t.role.value == "user" else "Assistant"
        body   = t.content
        if t.media_description:
            body = f"[Media fact: {t.media_description}]\n{body}".strip()
        lines.append(f"{prefix}: {body}")
    return "\n".join(lines)