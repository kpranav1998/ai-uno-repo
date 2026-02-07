"""RAG Pipeline orchestrator.

Wires together chunker, embedder, and retriever into a
single pipeline with ingest and query methods.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

from chunker import load_and_chunk
from embedder import embed_query, embed_texts
from retriever import VectorStore

EMBEDDINGS_FILE = "embeddings.npz"
CHUNKS_FILE = "chunks.json"


class RAGPipeline:
    """End-to-end retrieval pipeline for email search."""

    def __init__(self, data_dir: str = "."):
        """Initialize the pipeline.

        Args:
            data_dir: directory where embeddings and chunks cache are stored.
        """
        self.data_dir = data_dir
        self.vector_store = None

    def ingest(self, emails_dir: str) -> int:
        """Load, chunk, and embed all emails. Cache results to disk.

        Args:
            emails_dir: path to the directory containing email .txt files.

        Returns:
            Number of emails processed.
        """
        logger.info("Loading emails from %s...", emails_dir)
        chunks = load_and_chunk(emails_dir)
        logger.info("Loaded %d emails.", len(chunks))

        logger.info("Generating embeddings...")
        texts = [c["chunk_text"] for c in chunks]
        try:
            embeddings = embed_texts(texts)
        except Exception:
            logger.exception("Failed to generate embeddings")
            raise
        logger.info("Generated embeddings with shape %s.", embeddings.shape)

        # Save to disk
        emb_path = os.path.join(self.data_dir, EMBEDDINGS_FILE)
        chunks_path = os.path.join(self.data_dir, CHUNKS_FILE)

        try:
            np.savez_compressed(emb_path, embeddings=embeddings)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
        except OSError:
            logger.exception("Failed to save cache to %s", self.data_dir)
            raise

        logger.info("Saved embeddings to %s", emb_path)
        logger.info("Saved chunks to %s", chunks_path)

        self.vector_store = VectorStore(embeddings, chunks)
        return len(chunks)

    def load(self) -> bool:
        """Load cached embeddings and chunks from disk.

        Returns:
            True if successfully loaded, False if cache not found.
        """
        emb_path = os.path.join(self.data_dir, EMBEDDINGS_FILE)
        chunks_path = os.path.join(self.data_dir, CHUNKS_FILE)

        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            return False

        try:
            data = np.load(emb_path)
            embeddings = data["embeddings"]

            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except (OSError, json.JSONDecodeError, KeyError):
            logger.exception("Failed to load cache from %s", self.data_dir)
            return False

        self.vector_store = VectorStore(embeddings, chunks)
        logger.info("Loaded %d cached embeddings.", len(chunks))
        return True

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """Run retrieval: embed question, find top-k similar chunks.

        Args:
            question: the user's search query.
            top_k: number of chunks to retrieve.

        Returns:
            List of retrieved chunks with scores.
        """
        if self.vector_store is None:
            raise RuntimeError(
                "No embeddings loaded. Run ingest first or ensure cache exists."
            )

        try:
            query_emb = embed_query(question)
        except Exception:
            logger.exception("Failed to embed query: %s", question)
            raise

        return self.vector_store.retrieve(query_emb, top_k=top_k)
