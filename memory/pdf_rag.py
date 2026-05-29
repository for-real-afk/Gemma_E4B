"""
memory/pdf_rag.py
──────────────────
Lightweight retrieval-augmented generation for PDF content.

No vector DB required. Uses keyword overlap (BM25-style) to rank
chunks by relevance to the user's query, then assembles the top
chunks in original document order up to a configurable token budget.

Flow:
  1. chunk_text()              - split extracted PDF text into overlapping chunks
  2. retrieve_relevant_chunks() - rank by keyword overlap, return top chunks
                                  up to max_tokens, in document order
"""

import re
from typing import List

# Conservative estimate: 4 characters per token (works for English + code)
_CHARS_PER_TOKEN = 4

# Common words that carry no retrieval signal
_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "but", "for", "with", "this", "that", "what", "how", "are",
    "was", "were", "be", "been", "have", "has", "do", "does", "did",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "from", "by", "as", "if", "so", "not", "no", "can", "will",
    "would", "could", "should", "may", "might", "its", "their", "our",
    "your", "my", "which", "who", "when", "where", "then", "than",
}


def chunk_text(
    text: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks by approximate token count.

    Parameters
    ----------
    text          : raw extracted PDF text
    chunk_tokens  : target size of each chunk in tokens (~300 = ~1200 chars)
    overlap_tokens: token overlap between adjacent chunks to avoid cutting context

    Returns
    -------
    List of text chunks in document order.
    """
    words = text.split()
    if not words:
        return []

    # ~1.25 words per token (average 5-char word, 4 chars/token)
    words_per_chunk   = max(10, int(chunk_tokens * 1.25))
    words_per_overlap = max(1,  int(overlap_tokens * 1.25))
    step = words_per_chunk - words_per_overlap

    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), step)
        if words[i : i + words_per_chunk]
    ]


def _keyword_score(chunk: str, query: str) -> float:
    """
    Fraction of meaningful query words found in the chunk.
    Returns 0.0–1.0.
    """
    query_words = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    if not query_words:
        return 0.0
    chunk_words = set(re.findall(r"\w+", chunk.lower()))
    return len(query_words & chunk_words) / len(query_words)


def retrieve_relevant_chunks(
    chunks: List[str],
    query: str,
    max_tokens: int = 3000,
) -> str:
    """
    Rank chunks by keyword overlap with the query, select the top ones
    within the token budget, then return them in original document order
    (so the LLM sees coherent text, not a random jumble).

    Falls back to the document start when no chunk scores above zero
    (e.g. very generic query like "summarise this document").

    Parameters
    ----------
    chunks     : output of chunk_text()
    query      : the user's message / question
    max_tokens : maximum tokens of context to return (default 3000)

    Returns
    -------
    Assembled context string ready to inject into the prompt.
    """
    if not chunks:
        return ""

    max_chars = max_tokens * _CHARS_PER_TOKEN

    # ── Score every chunk ─────────────────────────────────────────────────────
    scored = sorted(
        enumerate(chunks),
        key=lambda t: _keyword_score(t[1], query),
        reverse=True,
    )

    # ── Greedily select until budget exhausted ────────────────────────────────
    selected: set[int] = set()
    used_chars = 0

    for idx, chunk in scored:
        if used_chars + len(chunk) > max_chars:
            # Try smaller remaining chunks before giving up
            continue
        selected.add(idx)
        used_chars += len(chunk)
        if used_chars >= max_chars * 0.95:   # 95% filled — stop
            break

    # ── Fallback: no keyword overlap — use document start ─────────────────────
    if not selected:
        for i, chunk in enumerate(chunks):
            if used_chars + len(chunk) > max_chars:
                break
            selected.add(i)
            used_chars += len(chunk)

    # ── Re-assemble in original document order ────────────────────────────────
    return "\n\n".join(chunks[i] for i in sorted(selected))
