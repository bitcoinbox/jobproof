# Project — Retrieval-Augmented Support Assistant

> Fictional sample project for the persona "Robin Vega."

## What it is

A support assistant at Northwind Labs that answers customer questions grounded in the
company's documentation and past resolved tickets, instead of free-associating. Live in
production in front of customers.

## How it works

- **Ingestion:** docs and resolved tickets are chunked (heading-aware, with overlap) and
  embedded into a vector store. Started on **Chroma** locally, then migrated the
  production index to **pgvector** on Postgres so retrieval lived next to the app's data.
- **Retrieval:** each incoming question is embedded and the top-k most relevant chunks are
  retrieved and passed to **the LLM** as grounding context, with citations back to the
  source doc/ticket.
- **Generation:** the LLM answers from the retrieved context only; if nothing relevant is
  retrieved, it says so and hands off to a human rather than guessing.
- **Guardrails:** the system prompt enforces "answer only from retrieved context," and
  every answer links its sources.

## My role

Designed the chunking + retrieval pipeline, the Chroma→pgvector migration, and the
grounding prompt. Wrote the offline retrieval tests that assert relevant chunks rank
first.

## Impact / lessons

- RAG cut "made-up answer" complaints sharply because the assistant cites sources.
- Learned the real tradeoff: retrieval shines when the corpus is large; for a tiny corpus
  it can underperform just sending everything. Measure, don't assume.

## Keywords

RAG, retrieval-augmented generation, vector database, Chroma, pgvector, embeddings,
chunking, LLM API, grounding, citations, Python, Postgres.
