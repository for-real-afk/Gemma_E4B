"""
memory/embeddings.py
─────────────────────
Ollama embedding API wrapper.

Uses /api/embed (batch endpoint, Ollama 0.1.26+).
Returns None on any failure so callers can fall back gracefully.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


async def embed_texts(
    texts:       list[str],
    ollama_host: str,
    model:       str,
    api_key:     str = "",
) -> Optional[list[list[float]]]:
    """
    Embed a batch of texts via Ollama /api/embed.

    Returns
    -------
    list of embedding vectors (one per input text), or None on failure.
    None triggers keyword-search fallback in the caller.
    """
    if not texts:
        return []

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{ollama_host}/api/embed",
                json={"model": model, "input": texts},
                headers=headers,
            )
    except Exception as exc:
        logger.warning("Ollama embed request failed: %s", exc)
        return None

    if r.status_code != 200:
        logger.warning(
            "Ollama embed returned HTTP %d — model '%s' may not be pulled. "
            "Run: ollama pull %s  →  falling back to keyword search.",
            r.status_code, model, model,
        )
        return None

    data = r.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        logger.warning("Ollama embed response missing 'embeddings' field.")
        return None

    logger.debug("Embedded %d texts via Ollama (%s)", len(texts), model)
    return embeddings


async def embed_query(
    text:        str,
    ollama_host: str,
    model:       str,
    api_key:     str = "",
) -> Optional[list[float]]:
    """Embed a single query text. Returns None on failure."""
    result = await embed_texts([text], ollama_host, model, api_key)
    return result[0] if result else None
