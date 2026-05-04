"""
memory/models.py
────────────────
Shared data models for the two-layer memory system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Turn:
    """
    A single conversation turn.

    For multimodal input the raw image is NEVER stored.
    Instead, `media_description` holds the text-semantic extracted
    from the image before it entered memory.
    """
    role:              Role
    content:           str
    media_description: Optional[str] = None   # extracted facts, not raw image

    def to_llm_message(self) -> dict:
        """Return the turn in the {role, content} format the LLM expects."""
        text = self.content
        if self.media_description:
            text = f"{self.media_description}\n\n{text}".strip()
        return {"role": self.role.value, "content": text}


@dataclass
class SessionMemory:
    """
    Full memory state for one session.

    short_term  — last N turns (verbatim, authoritative)
    summary     — compressed older context (background only)
    """
    session_id: str
    short_term: list[Turn] = field(default_factory=list)
    summary:    str        = ""                          # 150-300 tokens target
    turn_count: int        = 0                           # total turns ever seen