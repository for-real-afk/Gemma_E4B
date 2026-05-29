"""
memory/vector_store.py
───────────────────────
In-memory vector store backed by numpy cosine similarity.

No external service required. Lost on server restart — acceptable for demo.
When nomic-embed-text is available on Ollama, this gives semantic search.
When it isn't, callers fall back to keyword matching transparently.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorStore:
    """
    Stores pre-normalized chunk embeddings for a single session.
    Supports cosine similarity search via matrix multiplication.
    """
    chunks:      list[str]           = field(default_factory=list)
    _embeddings: Optional[np.ndarray] = field(default=None, repr=False)

    def add(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        """Store chunks and their pre-normalized embeddings."""
        self.chunks = chunks
        arr = np.array(embeddings, dtype=np.float32)
        # Pre-normalize once so every search is just a dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        self._embeddings = arr / (norms + 1e-8)
        logger.info("VectorStore: indexed %d chunks (dim=%d)", len(chunks), arr.shape[1])

    def search(
        self,
        query_embedding: list[float],
        top_k:           int = 6,
        max_tokens:      int = 2000,
    ) -> str:
        """
        Return the top-k most similar chunks concatenated in document order,
        trimmed to approximately max_tokens worth of text.

        Parameters
        ----------
        query_embedding : embedding of the user's question
        top_k           : max chunks to retrieve before token trimming
        max_tokens      : approximate token budget (4 chars ≈ 1 token)

        Returns
        -------
        Assembled context string, or "" if the store is empty.
        """
        if self._embeddings is None or not self.chunks:
            return ""

        q = np.array(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        scores      = self._embeddings @ q          # cosine similarity for all chunks
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Reassemble in original document order for coherent reading
        selected_indices = sorted(top_indices.tolist())

        max_chars = max_tokens * 4
        parts:    list[str] = []
        used:     int       = 0

        for i in selected_indices:
            chunk = self.chunks[i]
            if used + len(chunk) > max_chars:
                break
            parts.append(chunk)
            used += len(chunk)

        return "\n\n".join(parts)

    @property
    def ready(self) -> bool:
        return self._embeddings is not None and len(self.chunks) > 0


# ── session registry ──────────────────────────────────────────────────────────

_stores: dict[str, VectorStore] = {}


def get_store(session_id: str) -> Optional[VectorStore]:
    return _stores.get(session_id)


def set_store(session_id: str, store: VectorStore) -> None:
    _stores[session_id] = store


def delete_store(session_id: str) -> None:
    _stores.pop(session_id, None)
