"""Hybrid retrieval using cosine similarity + keyword + metadata matching.

Combines semantic embedding search with BM25-style keyword scoring
and direct metadata field matching so that name/metadata queries
work alongside topic-based queries.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


class VectorStore:
    """In-memory vector store with hybrid semantic + keyword retrieval."""

    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):
        """Initialize with pre-computed embeddings and their corresponding chunks.

        Args:
            embeddings: np.ndarray of shape (n_docs, embed_dim).
            chunks: list of dicts with keys chunk_text and metadata.

        Raises:
            ValueError: If embeddings and chunks have mismatched lengths or are empty.
        """
        if len(embeddings) == 0 or len(chunks) == 0:
            raise ValueError("Cannot create VectorStore with empty embeddings or chunks")
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) length mismatch"
            )

        self.embeddings = embeddings
        self.chunks = chunks
        # Pre-compute norms for cosine similarity
        self.norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Pre-compute BM25 components
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: Counter[str, int] = Counter()
        self._avg_dl = 0.0
        self._build_keyword_index()

    def _build_keyword_index(self) -> None:
        """Build token lists and document-frequency counts for BM25."""
        for chunk in self.chunks:
            tokens = _tokenize(chunk["chunk_text"])
            self._doc_tokens.append(tokens)
            unique = set(tokens)
            for t in unique:
                self._doc_freqs[t] += 1

        total_tokens = sum(len(t) for t in self._doc_tokens)
        self._avg_dl = total_tokens / max(len(self._doc_tokens), 1)
        logger.info(
            "Built keyword index: %d docs, %d unique tokens, avg doc length %.1f",
            len(self._doc_tokens),
            len(self._doc_freqs),
            self._avg_dl,
        )

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        """Compute BM25 scores for all documents against the query."""
        query_tokens = _tokenize(query)
        n = len(self.chunks)
        scores = np.zeros(n)

        for qt in query_tokens:
            df = self._doc_freqs.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc_tokens in enumerate(self._doc_tokens):
                tf = doc_tokens.count(qt)
                dl = len(doc_tokens)
                denom = tf + k1 * (1 - b + b * dl / self._avg_dl)
                scores[i] += idf * (tf * (k1 + 1)) / denom

        return scores

    def _metadata_scores(self, query: str) -> np.ndarray:
        """Score documents by direct metadata field matching.

        Checks query tokens against subject, from, and to fields.
        Returns 1.0 for exact substring match, partial credit for
        token overlap, 0.0 for no match.
        """
        query_lower = query.lower()
        query_tokens = set(_tokenize(query))
        n = len(self.chunks)
        scores = np.zeros(n)

        for i, chunk in enumerate(self.chunks):
            meta = chunk["metadata"]
            best = 0.0
            for field in ("subject", "from", "to"):
                value = meta.get(field, "").lower()
                if not value:
                    continue
                # Full substring match (e.g. "helen powell" in "Helen Powell <helen...>")
                if query_lower in value or value in query_lower:
                    best = max(best, 1.0)
                    continue
                # Token overlap: fraction of query tokens found in field
                field_tokens = set(_tokenize(value))
                if not query_tokens:
                    continue
                overlap = len(query_tokens & field_tokens) / len(query_tokens)
                best = max(best, overlap)

            scores[i] = best

        return scores

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_text: str = "",
        top_k: int = 5,
        semantic_weight: float = 0.70,
        keyword_weight: float = 0.15,
        metadata_weight: float = 0.15,
    ) -> list[dict]:
        """Find the top-k chunks using hybrid semantic + keyword + metadata scoring.

        Final score = semantic_weight * cosine + keyword_weight * BM25 + metadata_weight * meta

        Args:
            query_embedding: np.ndarray of shape (embed_dim,).
            query_text: raw query string for keyword and metadata matching.
            top_k: number of results to return.
            semantic_weight: weight for cosine similarity.
            keyword_weight: weight for BM25 keyword score.
            metadata_weight: weight for metadata field matching.

        Returns:
            List of dicts with keys: chunk_text, metadata, score.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            logger.warning("Query embedding has zero norm; results may be meaningless")

        cosine_scores = (self.embeddings @ query_embedding) / (
            self.norms.squeeze() * max(query_norm, 1e-10)
        )

        if query_text:
            bm25_raw = self._bm25_scores(query_text)
            bm25_max = bm25_raw.max()
            bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

            meta_scores = self._metadata_scores(query_text)

            combined = (
                semantic_weight * cosine_scores
                + keyword_weight * bm25_norm
                + metadata_weight * meta_scores
            )
            logger.debug(
                "Query '%s' — top cosine=%.3f, top bm25=%.3f, top meta=%.3f",
                query_text,
                float(cosine_scores.max()),
                float(bm25_norm.max()),
                float(meta_scores.max()),
            )
        else:
            combined = cosine_scores

        top_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_text": self.chunks[idx]["chunk_text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(combined[idx]),
            })

        logger.info(
            "Retrieved %d results for query '%s' (top score: %.3f)",
            len(results),
            query_text or "<embedding-only>",
            results[0]["score"] if results else 0.0,
        )
        return results
