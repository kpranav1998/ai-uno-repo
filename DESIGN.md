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

**Approach**: Brute-force cosine similarity using numpy.

For 100 documents with 384-dimensional embeddings, exact search is instant (<1ms). The vector store pre-computes document norms at initialization to avoid redundant computation during queries.

```
similarity = (doc_embeddings @ query_embedding) / (doc_norms * query_norm)
```

No approximate nearest neighbor (ANN) index is needed at this scale. FAISS or Annoy would add complexity with no measurable benefit under ~10K documents.

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
| Brute-force cosine similarity | Simple and exact, but O(n) per query — won't scale past ~100K docs |
| No re-ranking | Embedding similarity alone may miss nuanced relevance |
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

# Retrieve more/fewer chunks
python main.py --top-k 3
```
