# Project — Prompt-Cached Batch Service

> Fictional sample project for the persona "Robin Vega."

## What it is

A batch service that runs the same large, stable instruction + context prefix across many
items (classifying a queue of documents), made cheap with **prompt caching**.

## How it works

- The big stable prefix (system instructions + shared reference context) is marked as a
  cache breakpoint; only the per-item content changes after it.
- The first request writes the cache; every subsequent item reads the prefix from cache at
  a fraction of the input cost.
- Cache hits are tracked via `cache_read_input_tokens` and surfaced in logs so cost is
  observable, not guessed.

## My role

Designed the prompt layout so the stable prefix came first (and stayed byte-identical),
verified cache hits, and wired the per-item loop. Containerized it and ran it on Railway.

## Impact / lessons

- Cut per-request input cost substantially on high-volume batches by reusing the cached
  prefix.
- Learned the discipline of prompt caching: keep the prefix stable (no timestamps/UUIDs up
  front), put volatile content last, and *verify* hits — a silent invalidator quietly
  doubles your bill.

## Keywords

prompt caching, cost optimization, batch processing, LLM API, Docker, Railway,
observability, Python.
