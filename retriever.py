"""Vector retrieval using cosine similarity with numpy.

Stores document embeddings and performs similarity search against queries.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store backed by a numpy array."""

    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):
        """Initialize with pre-computed embeddings and their corresponding chunks.

        Args:
            embeddings: np.ndarray of shape (n_docs, embed_dim).
            chunks: list of dicts with keys chunk_text and metadata.
        """
        self.embeddings = embeddings
        self.chunks = chunks
        # Pre-compute norms for cosine similarity
        self.norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Find the top-k most similar chunks to the query.

        Args:
            query_embedding: np.ndarray of shape (embed_dim,).
            top_k: number of results to return.

        Returns:
            List of dicts with keys: chunk_text, metadata, score.
        """
        query_norm = np.linalg.norm(query_embedding)
        # Cosine similarity: dot(query, doc) / (|query| * |doc|)
        similarities = (self.embeddings @ query_embedding) / (
            self.norms.squeeze() * query_norm
        )

        # Get top-k indices (descending similarity)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_text": self.chunks[idx]["chunk_text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(similarities[idx]),
            })

        return results
