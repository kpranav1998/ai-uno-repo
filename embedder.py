"""Embedding generation using sentence-transformers.

Uses the all-MiniLM-L6-v2 model (384-dim) for fast, quality embeddings.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME)
        except Exception:
            logger.exception("Failed to load embedding model '%s'", MODEL_NAME)
            raise
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch embed a list of texts.

    Returns np.ndarray of shape (len(texts), 384).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Returns np.ndarray of shape (384,).
    """
    model = _get_model()
    embedding = model.encode(query, convert_to_numpy=True)
    return embedding
