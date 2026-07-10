# Bullet Variants — vocabulary bank

> Fictional sample for the persona "Robin Vega." Pre-written bullets in several
> vocabularies so the tailorer can mirror a posting's exact wording (GenAI vs LLM vs
> agentic, etc.) without inventing anything.

## LLM integration

- Integrated **LLM APIs** (Anthropic, OpenAI) into production features with tuned
  system prompts and **structured outputs**.
- Shipped **GenAI** features used by real customers — a chat assistant and an automation
  pipeline — not demos.
- Built **agentic** workflows using **tool use / function calling** to turn unstructured
  input into validated structured records.

## Retrieval / RAG

- Built **retrieval-augmented generation (RAG)** over a document corpus with a **vector
  database** (Chroma, then pgvector) and **embedding-based retrieval**.
- Designed heading-aware **chunking** with overlap and citation-grounded answers.

## Evaluation

- Built an **evaluation harness** measuring keyword/field match rate and tracking quality
  across prompt/model changes, gated in **CI**.
- Ran **A/B** comparisons (RAG vs. no-RAG) to quantify changes instead of guessing.

## Cost / performance

- Designed a **prompt-cached** LLM pipeline that reuses a stable context prefix to cut
  per-request input cost; verified cache hits in production.

## Platform / ops

- Built and operated **FastAPI** services backed by **PostgreSQL** and **Redis queues**,
  containerized with **Docker** and deployed via **CI/CD** to Railway/Vercel.
- Owned services end to end: design → tests → deploy → on-call.
