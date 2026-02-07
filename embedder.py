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
            logger.info("Loading embedding model '%s'...", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except Exception:
            logger.exception("Failed to load embedding model '%s'", MODEL_NAME)
            raise
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch embed a list of texts.

    Returns np.ndarray of shape (len(texts), 384).

    Raises:
        ValueError: If texts list is empty.
        RuntimeError: If encoding fails.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")
    logger.info("Embedding %d texts...", len(texts))
    try:
        model = _get_model()
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    except ValueError:
        raise
    except Exception:
        logger.exception("Failed to embed %d texts", len(texts))
        raise RuntimeError(f"Embedding failed for batch of {len(texts)} texts") from None
    logger.info("Finished embedding %d texts.", len(texts))
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string.

    Returns np.ndarray of shape (384,).

    Raises:
        ValueError: If query is empty.
        RuntimeError: If encoding fails.
    """
    if not query or not query.strip():
        raise ValueError("Cannot embed an empty query")
    try:
        model = _get_model()
        embedding = model.encode(query, convert_to_numpy=True)
    except ValueError:
        raise
    except Exception:
        logger.exception("Failed to embed query: %s", query)
        raise RuntimeError(f"Embedding failed for query: {query!r}") from None
    return embedding
