# Mini RAG System

```
Time: 75 Minutes
```

## Problem

Build a Retrieval-Augmented Generation (RAG) pipeline that can process documents, retrieve relevant information, and generate answers to questions.

## Dataset

A dataset of 100 synthetic emails is provided in the `emails/` directory. Each email contains:
- Subject line
- Sender and receiver information (from a pool of 200 unique people)
- Body content (100+ words) with diverse topics including:
  - Project updates
  - Meeting requests
  - Budget approvals
  - Technical issues
  - Client feedback
  - Team announcements
  - Deadline extensions
  - Training opportunities
  - Vendor proposals
  - Performance reviews

Use this dataset to test your RAG system's ability to:
- Chunk and process email documents
- Retrieve relevant information based on queries
- Generate accurate answers using the retrieved context

## Requirements

Your RAG system should implement:

1. **Document Chunking**
   - Split documents into appropriate chunks
   - Handle different document types and sizes
   - Explain your chunking strategy

2. **Embedding**
   - Generate embeddings for document chunks
   - Choose an appropriate embedding model
   - Store embeddings efficiently

3. **Retrieval**
   - Implement similarity search to find relevant chunks
   - Handle query embedding and matching
   - Return top-k most relevant results

4. **Generation**
   - Use retrieved context to generate answers
   - Design effective prompts
   - Integrate with a language model

## Constraints

- Do not use end-to-end RAG frameworks (e.g., LangChain, LlamaIndex)
- Build core components yourself or use individual libraries
- Document your design choices and tradeoffs
- Explain your approach to quality evaluation

## Submission

- Create a public git repository containing your submission and share the repository link
- Do not fork this repository or create pull requests
