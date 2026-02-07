"""CLI entry point for the Mini RAG System.

Usage:
    python main.py --ingest              Process emails and generate embeddings
    python main.py                       Interactive query mode (uses cached embeddings)
    python main.py --query "some text"   Single query mode (non-interactive)
    python main.py --top-k 3             Change number of retrieved chunks (default: 5)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

EMAILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails")


def display_results(retrieved: list[dict]) -> None:
    """Pretty-print retrieved chunks to stdout."""
    print(f"\n{'=' * 60}")
    print(f"Results ({len(retrieved)} retrieved):")
    print(f"{'=' * 60}")
    for i, result in enumerate(retrieved, 1):
        meta = result["metadata"]
        print(
            f"\n--- {i}. [{meta['filename']}] (score: {result['score']:.3f}) ---"
        )
        print(f"Subject: {meta['subject']}")
        print(f"From: {meta['from']}")
        print(f"To: {meta['to']}")
        body = result["chunk_text"].split("\n\n", 1)[-1]
        snippet = body[:200] + ("..." if len(body) > 200 else "")
        print(f"Preview: {snippet}")
    print()


def run_single_query(pipeline: RAGPipeline, question: str, top_k: int) -> int:
    """Execute a single query and display results. Returns exit code."""
    try:
        retrieved = pipeline.query(question, top_k=top_k)
    except RuntimeError as e:
        logger.error("Query failed: %s", e)
        return 1
    except Exception:
        logger.exception("Unexpected error during query: %s", question)
        return 1

    display_results(retrieved)
    return 0


def run_interactive(pipeline: RAGPipeline, top_k: int) -> None:
    """Run the interactive query loop."""
    print("\nMini RAG System - Email Search")
    print("Type your query and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            retrieved = pipeline.query(question, top_k=top_k)
        except RuntimeError as e:
            logger.error("Query failed: %s", e)
            continue
        except Exception:
            logger.exception("Unexpected error during query: %s", question)
            continue

        display_results(retrieved)


def main() -> int:
    """Entry point. Returns exit code."""
    parser = argparse.ArgumentParser(description="Mini RAG System for email Q&A")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Process emails and generate embeddings",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query (non-interactive)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        pipeline = RAGPipeline(data_dir=data_dir)
    except Exception:
        logger.exception("Failed to initialize pipeline")
        return 1

    # --- Ingest mode ---
    if args.ingest:
        if not os.path.isdir(EMAILS_DIR):
            logger.error("Emails directory not found at %s", EMAILS_DIR)
            return 1
        try:
            count = pipeline.ingest(EMAILS_DIR)
        except Exception:
            logger.exception("Ingestion failed")
            return 1
        print(f"\nIngestion complete. Processed {count} emails.")
        print("You can now run 'python main.py' to start querying.")
        return 0

    # --- Query modes (need cached embeddings) ---
    try:
        loaded = pipeline.load()
    except Exception:
        logger.exception("Failed to load cached embeddings")
        return 1

    if not loaded:
        logger.error("No cached embeddings found. Run with --ingest first.")
        return 1

    # Single query mode
    if args.query:
        return run_single_query(pipeline, args.query, args.top_k)

    # Interactive mode
    run_interactive(pipeline, args.top_k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
