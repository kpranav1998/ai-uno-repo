# Mini RAG System - Design Document

## Architecture Overview

```
emails/ → Chunker → Embedder → Vector Store (numpy)
                                     ↓
User Query → Query Embedding → Similarity Search → Top-K Chunks → Display
```

The system is a from-scratch retrieval pipeline built without frameworks like LangChain or LlamaIndex. It processes 100 synthetic corporate emails and retrieves the most relevant ones for any natural language query using semantic similarity.

## Component Design

### Chunking (`chunker.py`)

**Strategy**: One chunk per email, with metadata embedded as a text prefix.

Each email is ~150 words — well within the embedding model's optimal window (~256 tokens). Splitting emails into smaller chunks would break context: separating a subject line from its body would make it impossible to answer "Who sent emails about budget approvals?" since the subject and sender would be in different chunks.

The chunk format preserves full structure:
```
Subject: {subject}
From: {from}
To: {to}

{body}
```

This means a query like "emails from Helen Powell" will have high cosine similarity with chunks containing her name in the From: field.

### Embeddings (`embedder.py`)

**Model**: `all-MiniLM-L6-v2` (384 dimensions)

- Fast inference (~100 docs/sec on CPU)
- Good quality for semantic similarity tasks
- Small model size (~80MB) — practical for local use
- 256-token input sweet spot aligns with our ~150-word emails

The model is lazy-loaded (initialized on first use) to avoid startup cost when only loading cached embeddings.

### Retrieval (`retriever.py`)

**Approach**: Hybrid retrieval combining three scoring signals.

Pure embedding similarity struggles with name/metadata queries (e.g. "emails from Helen Powell") because the model prioritizes semantic content over proper nouns. To solve this, retrieval combines three signals:

```
final_score = 0.70 × cosine_similarity + 0.15 × BM25_keyword + 0.15 × metadata_match
```

1. **Cosine similarity** (semantic) — embedding-based similarity for topic and meaning
2. **BM25 keyword scoring** — TF-IDF-style token matching for exact term relevance
3. **Metadata field matching** — direct substring and token overlap against `from`, `to`, and `subject` fields

The metadata scorer gives full credit (1.0) for exact substring matches and partial credit proportional to token overlap. This ensures "emails from Helen Powell" surfaces the correct results even when the embedding model doesn't rank them highly.

For 100 documents with 384-dimensional embeddings, exact search is instant (<1ms). No approximate nearest neighbor (ANN) index is needed at this scale.

### Pipeline (`rag_pipeline.py`)

The orchestrator provides two main operations:
- **`ingest()`**: Load → chunk → embed → cache to disk (embeddings.npz + chunks.json)
- **`query()`**: Embed question → retrieve top-k similar chunks

Embeddings are cached as compressed numpy arrays (.npz) to avoid re-computing on every session. At 100 emails × 384 dims × 4 bytes, the cache is ~150KB.

## Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| One chunk per email | Preserves full context but limits granularity for longer documents |
| Metadata as text prefix | Aids retrieval on sender/subject queries but slightly dilutes body semantics |
| Hybrid scoring (cosine + BM25 + metadata) | Handles both semantic and exact-match queries, but adds scoring complexity |
| Fixed weight split (0.70 / 0.15 / 0.15) | Semantic-heavy balance; could be tuned per query type with a classifier |
| Single embedding model | Same model for docs and queries; asymmetric models could improve accuracy |

## Quality Evaluation Approach

To evaluate retrieval quality:

1. **Manual spot checks**: For known queries ("Who sent emails about budget?"), verify the top-k results contain relevant emails
2. **Relevance scoring**: Check if the correct email appears in top-1, top-3, top-5 (Recall@K)

For a production system, you'd want:
- A labeled evaluation dataset (query → relevant email IDs)
- Automated metrics (MRR, NDCG, Recall@K)
- A/B testing of chunking strategies and embedding models

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest emails (generates embeddings)
python main.py --ingest

# Interactive search
python main.py

# Single query (non-interactive)
python main.py --query "emails from Helen Powell"

# Retrieve more/fewer chunks
python main.py --top-k 3

# Combine flags
python main.py --query "budget approval" --top-k 3
```
