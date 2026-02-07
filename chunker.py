"""Document chunking for email files.

Parses email .txt files, extracts metadata (subject, from, to) and body,
and produces one chunk per email with metadata preserved as prefix text.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def parse_email(filepath: str) -> dict:
    """Parse a single email file into structured fields.

    Returns dict with keys: subject, from, to, body, filename.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    subject_match = re.search(r"^Subject:\s*(.+)$", text, re.MULTILINE)
    from_match = re.search(r"^From:\s*(.+)$", text, re.MULTILINE)
    to_match = re.search(r"^To:\s*(.+)$", text, re.MULTILINE)

    subject = subject_match.group(1).strip() if subject_match else ""
    sender = from_match.group(1).strip() if from_match else ""
    recipient = to_match.group(1).strip() if to_match else ""

    if not subject_match:
        logger.warning("Missing Subject header in %s", filepath)
    if not from_match:
        logger.warning("Missing From header in %s", filepath)
    if not to_match:
        logger.warning("Missing To header in %s", filepath)

    # Body starts after the last header (To: line), skip blank lines
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("To:"):
            body_start = i + 1
            break

    body = "\n".join(lines[body_start:]).strip()

    return {
        "subject": subject,
        "from": sender,
        "to": recipient,
        "body": body,
        "filename": os.path.basename(filepath),
    }


def make_chunk(email: dict) -> str:
    """Create a single chunk string from a parsed email.

    Embeds metadata as prefix so the embedding captures sender/subject info.
    """
    return (
        f"Subject: {email['subject']}\n"
        f"From: {email['from']}\n"
        f"To: {email['to']}\n\n"
        f"{email['body']}"
    )


def load_and_chunk(emails_dir: str) -> list[dict]:
    """Load all emails from a directory and return chunks with metadata.

    Returns list of dicts, each with keys: chunk_text, metadata.
    """
    files = sorted(
        f for f in os.listdir(emails_dir) if f.endswith(".txt")
    )
    logger.info("Found %d email files in %s", len(files), emails_dir)

    chunks = []
    for filename in files:
        filepath = os.path.join(emails_dir, filename)
        try:
            email = parse_email(filepath)
        except Exception:
            logger.exception("Failed to parse email: %s", filepath)
            continue
        chunk_text = make_chunk(email)
        metadata = {
            "subject": email["subject"],
            "from": email["from"],
            "to": email["to"],
            "filename": email["filename"],
        }
        chunks.append({"chunk_text": chunk_text, "metadata": metadata})

    logger.info("Successfully chunked %d / %d emails.", len(chunks), len(files))
    return chunks
