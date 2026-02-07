"""CLI entry point for the Mini RAG System.

Usage:
    python main.py --ingest       Process emails and generate embeddings
    python main.py                Interactive query mode (uses cached embeddings)
    python main.py --top-k 3      Change number of retrieved chunks (default: 5)
"""

import argparse
import logging
import os
import sys

from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

EMAILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails")


def main():
    parser = argparse.ArgumentParser(description="Mini RAG System for email Q&A")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Process emails and generate embeddings",
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
    pipeline = RAGPipeline(data_dir=data_dir)

    if args.ingest:
        if not os.path.isdir(EMAILS_DIR):
            logger.error("Emails directory not found at %s", EMAILS_DIR)
            sys.exit(1)
        count = pipeline.ingest(EMAILS_DIR)
        print(f"\nIngestion complete. Processed {count} emails.")
        print("You can now run 'python main.py' to start querying.")
        return

    # Try to load cached embeddings
    if not pipeline.load():
        logger.error("No cached embeddings found. Run with --ingest first.")
        sys.exit(1)

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
            retrieved = pipeline.query(question, top_k=args.top_k)
        except Exception:
            logger.exception("Query failed for input: %s", question)
            continue

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
            # Show a snippet of the body (first 200 chars)
            body = result["chunk_text"].split("\n\n", 1)[-1]
            snippet = body[:200] + ("..." if len(body) > 200 else "")
            print(f"Preview: {snippet}")
        print()


if __name__ == "__main__":
    main()
